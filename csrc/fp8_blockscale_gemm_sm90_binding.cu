
#include <tvm/ffi/extra/module.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "tvm_ffi_utils.h"

namespace kernels = tensorrt_llm::kernels::fp8_blockscale_gemm;

using tvm::ffi::Function;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;

#ifdef FLASHINFER_ENABLE_FP8_E4M3
inline bool is_fp8_e4m3fn(DLDataType dtype) {
  return encode_dlpack_dtype(dtype) == float8_e4m3fn_code;
}
#else
inline bool is_fp8_e4m3fn(DLDataType) { return false; }
#endif

// Shared validation for the grouped MoE GEMM entry points. Offset values are
// checked device-side (fused kernel prologue / preflight kernel below): a
// host-side read would be a stream sync and break CUDA-graph capture.
inline void check_moe_offsets(const char* what, TensorView offsets, TensorView a) {
  CHECK_INPUT(offsets);  // CUDA-resident + contiguous (kernels index it raw)
  auto od = offsets.dtype();
  TVM_FFI_ICHECK(od.code == kDLInt && od.bits == 64 && od.lanes == 1)
      << what << ": offsets must be int64";
  TVM_FFI_ICHECK(offsets.ndim() == 1) << what << ": offsets must be 1D";
  TVM_FFI_ICHECK(offsets.size(0) >= 2) << what << ": offsets needs >= 2 entries (G+1)";
  CHECK_DEVICE(offsets, a);
}

inline void check_moe_shape_scalars(const char* what, int64_t shape_n, int64_t shape_k) {
  TVM_FFI_ICHECK(shape_n > 0 && shape_k > 0)
      << what << ": shapes must be positive, got N=" << shape_n << " K=" << shape_k;
}

// Same-stream preflight for checked plain entry calls (graph-safe, no host
// sync): traps on malformed offsets and on a final offset beyond the D-row
// or A-side capacity.
__global__ void moe_offsets_preflight_kernel(const int64_t* __restrict__ offsets, int num_problems,
                                             int64_t row_capacity, int64_t a_capacity) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int64_t prev = offsets[0];
  bool bad = prev != 0;
  for (int g = 1; g <= num_problems && !bad; ++g) {
    int64_t cur = offsets[g];
    bad = cur < prev;
    prev = cur;
  }
  if (bad || prev > row_capacity || prev > a_capacity) {
    printf(
        "moe_gemm: bad offsets (first != 0, decreasing, or offsets[%d]=%lld > capacity "
        "min(D=%lld, A=%lld))\n",
        num_problems, static_cast<long long>(prev), static_cast<long long>(row_capacity),
        static_cast<long long>(a_capacity));
    asm volatile("trap;");
  }
}

class Fp8BlockScaleGemmRunner : public tvm::ffi::ModuleObj {
 public:
  Fp8BlockScaleGemmRunner() {
    // Instantiate runners for all supported combinations
    runner_bf16_bf16_ = std::make_unique<
        kernels::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>();

    runner_bf16_fp8_ = std::make_unique<
        kernels::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>();

    runner_fp8_fp8_ = std::make_unique<
        kernels::CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>>();
  }

  ~Fp8BlockScaleGemmRunner() = default;

  const char* type_key() const { return "flashinfer.Fp8BlockScaleGemmRunner"; }
  const char* kind() const final { return "fp8_blockscale_gemm_runner"; }

  Optional<Function> GetFunction(const tvm::ffi::String& name) {
    if (name == "run_gemm") {
      return Function::FromTyped([this](TensorView input, TensorView weight, TensorView output,
                                        Optional<TensorView> scales_a,
                                        Optional<TensorView> scales_b) {
        runGemm(input, weight, output, scales_a, scales_b);
      });
    } else if (name == "fp8_quantize_1x128") {
      return Function::FromTyped([this](TensorView input, TensorView outValueE4M3,
                                        TensorView outScaleFP8SF, bool use_ue8m0) {
        fp8_quantize_1x128(input, outValueE4M3, outScaleFP8SF, use_ue8m0);
      });
    } else if (name == "get_workspace_size") {
      return Function::FromTyped(
          [this](int64_t shape_m, int64_t shape_n, int64_t shape_k) -> int64_t {
            return getWorkspaceSize(shape_m, shape_n, shape_k);
          });
    } else if (name == "configure_workspace") {
      return Function::FromTyped([this](TensorView workspace) { configureWorkspace(workspace); });
    } else if (name == "moe_gemm") {
      return Function::FromTyped([this](TensorView d, TensorView a, TensorView b,
                                        TensorView offsets, int64_t shape_n, int64_t shape_k,
                                        Optional<TensorView> scales_a,
                                        Optional<TensorView> scales_b, bool trusted_offsets) {
        check_moe_shape_scalars("moe_gemm", shape_n, shape_k);
        check_moe_offsets("moe_gemm", offsets, a);
        size_t num_problems = static_cast<size_t>(offsets.size(0)) - 1;
        CHECK_INPUT(a);
        CHECK_INPUT(b);
        CHECK_INPUT(d);
        CHECK_DEVICE(b, a);
        CHECK_DEVICE(d, a);
        CHECK_DIM(2, a);
        TVM_FFI_ICHECK(a.size(1) == shape_k)
            << "moe_gemm: a is (M, K); a.size(1)=" << a.size(1) << " != K=" << shape_k;
        bool a_fp8 = is_fp8_e4m3fn(a.dtype());
        bool b_fp8 = is_fp8_e4m3fn(b.dtype());
        TVM_FFI_ICHECK(a_fp8 || a.dtype() == dl_bfloat16)
            << "moe_gemm: a must be float8_e4m3fn or bfloat16";
        TVM_FFI_ICHECK(b_fp8 || b.dtype() == dl_bfloat16)
            << "moe_gemm: b must be float8_e4m3fn or bfloat16";
        // b: (G, N, K) or the flattened (G*N, K) view
        if (b.ndim() == 3) {
          TVM_FFI_ICHECK(b.size(0) == static_cast<int64_t>(num_problems) && b.size(1) == shape_n &&
                         b.size(2) == shape_k)
              << "moe_gemm: b must be (G=" << num_problems << ", N=" << shape_n << ", K=" << shape_k
              << "), got (" << b.size(0) << ", " << b.size(1) << ", " << b.size(2) << ")";
        } else {
          CHECK_DIM(2, b);
          TVM_FFI_ICHECK(b.size(0) == static_cast<int64_t>(num_problems) * shape_n &&
                         b.size(1) == shape_k)
              << "moe_gemm: 2D b must be (G*N, K)";
        }
        CHECK_DIM(2, d);
        TVM_FFI_ICHECK(d.dtype() == dl_bfloat16) << "moe_gemm: d must be bfloat16";
        TVM_FFI_ICHECK(d.size(1) == shape_n)
            << "moe_gemm: d must be (Mcap, N=" << shape_n << "), got d.size(1)=" << d.size(1);
        auto* runner = selectRunner(a_fp8, b_fp8);
        TVM_FFI_ICHECK(runner != nullptr) << "moe_gemm: unsupported dtype combination";
        checkMoeContract("moe_gemm", runner, num_problems, shape_n, shape_k, a);
        // TMA DECLARED-ROW CONSTRAINT: A and D descriptors are declared
        // with the workspace-frozen aligned row count and TMA clamps to
        // the DECLARATION -- buffers must cover it, not just live rows.
        int64_t a_rows_decl = runner->getAlignedARows();
        TVM_FFI_ICHECK(a_rows_decl > 0) << "moe_gemm: runner has no frozen A-row state; call "
                                           "get_moe_workspace_size first";
        TVM_FFI_ICHECK(d.size(0) >= a_rows_decl)
            << "moe_gemm: d has " << d.size(0) << " rows, needs >= the frozen TMA "
            << "declaration " << a_rows_decl;
        if (a_fp8) {
          TVM_FFI_ICHECK(a.size(0) >= a_rows_decl)
              << "moe_gemm: a has " << a.size(0) << " rows, needs >= the frozen TMA "
              << "declaration " << a_rows_decl;
          TVM_FFI_ICHECK(scales_a.has_value()) << "moe_gemm: pre-quantized fp8 a requires scales_a";
        }
        if (b_fp8) {
          TVM_FFI_ICHECK(scales_b.has_value()) << "moe_gemm: pre-quantized fp8 b requires scales_b";
        }
        if (scales_a.has_value()) {
          CHECK_INPUT(scales_a.value());
          CHECK_DEVICE(scales_a.value(), a);
          auto sad = scales_a.value().dtype();
          TVM_FFI_ICHECK(sad.code == kDLFloat && sad.bits == 32)
              << "moe_gemm: scales_a must be float32";
          if (a_fp8 && shape_k % 128 == 0) {
            int64_t p_stride = runner->getMoePaddedStride();
            TVM_FFI_ICHECK(scales_a.value().numel() >= (shape_k / 128) * p_stride)
                << "moe_gemm: scales_a has " << scales_a.value().numel()
                << " elements, needs >= (K/128=" << (shape_k / 128) << ") * (P=" << p_stride
                << ") = " << (shape_k / 128) * p_stride;
          }
        }
        if (scales_b.has_value()) {
          CHECK_INPUT(scales_b.value());
          CHECK_DEVICE(scales_b.value(), a);
          auto sbd = scales_b.value().dtype();
          TVM_FFI_ICHECK(sbd.code == kDLFloat && sbd.bits == 32)
              << "moe_gemm: scales_b must be float32";
          if (b_fp8 && shape_n % 128 == 0 && shape_k % 128 == 0) {
            TVM_FFI_ICHECK(scales_b.value().numel() >=
                           static_cast<int64_t>(num_problems) * (shape_n / 128) * (shape_k / 128))
                << "moe_gemm: scales_b too small for (G, N/128, K/128)";
          }
        }
        auto stream = get_stream(a.device());
        int64_t a_capacity = std::min(a.size(0), runner->getAlignedARows());
        if (!trusted_offsets) {
          moe_offsets_preflight_kernel<<<1, 32, 0, stream>>>(
              static_cast<int64_t const*>(offsets.data_ptr()), static_cast<int>(num_problems),
              d.size(0), a_capacity);
          auto launch_error = cudaGetLastError();
          TVM_FFI_ICHECK_EQ(launch_error, cudaSuccess)
              << "moe_gemm: offsets preflight launch failed: " << cudaGetErrorString(launch_error);
        }
        runner->moeGemm(
            d.data_ptr(), a.data_ptr(), b.data_ptr(),
            static_cast<int64_t const*>(offsets.data_ptr()), num_problems,
            static_cast<size_t>(shape_n), static_cast<size_t>(shape_k), stream,
            scales_a.has_value() ? reinterpret_cast<float const*>(scales_a.value().data_ptr())
                                 : nullptr,
            scales_b.has_value() ? reinterpret_cast<float const*>(scales_b.value().data_ptr())
                                 : nullptr);
      });
    } else if (name == "moe_gemm_fc1_fused") {
      return Function::FromTyped([this](TensorView d_fp8, TensorView sfa2, TensorView a,
                                        TensorView b, TensorView offsets,
                                        int64_t shape_n_interleaved, int64_t shape_k,
                                        TensorView scales_a, TensorView scales_b,
                                        bool trusted_offsets) {
        check_moe_shape_scalars("moe_gemm_fc1_fused", shape_n_interleaved, shape_k);
        TVM_FFI_ICHECK(shape_n_interleaved % 256 == 0)
            << "moe_gemm_fc1_fused: interleaved N (2I) must be a multiple of 256 "
            << "(I % 128 == 0), got " << shape_n_interleaved;
        TVM_FFI_ICHECK(shape_k % 128 == 0)
            << "moe_gemm_fc1_fused: K must be a multiple of 128, got " << shape_k;
        int64_t i_size = shape_n_interleaved / 2;

        check_moe_offsets("moe_gemm_fc1_fused", offsets, a);  // shared strict validator
        size_t num_problems = static_cast<size_t>(offsets.size(0)) - 1;

        CHECK_INPUT(a);
        CHECK_INPUT(b);
        CHECK_INPUT(scales_a);
        CHECK_INPUT(scales_b);
        TVM_FFI_ICHECK(is_fp8_e4m3fn(a.dtype()) && is_fp8_e4m3fn(b.dtype()))
            << "moe_gemm_fc1_fused: a and b must be pre-quantized float8_e4m3fn";
        CHECK_DIM(2, a);
        TVM_FFI_ICHECK(a.size(1) == shape_k)
            << "moe_gemm_fc1_fused: a is (M, K); a.size(1)=" << a.size(1) << " != K=" << shape_k;
        // b: interleaved w13, (G, 2I, K) or the flattened (G*2I, K) view
        if (b.ndim() == 3) {
          TVM_FFI_ICHECK(b.size(0) == static_cast<int64_t>(num_problems) &&
                         b.size(1) == shape_n_interleaved && b.size(2) == shape_k)
              << "moe_gemm_fc1_fused: b must be (G=" << num_problems
              << ", 2I=" << shape_n_interleaved << ", K=" << shape_k << "), got (" << b.size(0)
              << ", " << b.size(1) << ", " << b.size(2) << ")";
        } else {
          CHECK_DIM(2, b);
          TVM_FFI_ICHECK(b.size(0) == static_cast<int64_t>(num_problems) * shape_n_interleaved &&
                         b.size(1) == shape_k)
              << "moe_gemm_fc1_fused: 2D b must be (G*2I, K)";
        }
        auto sad = scales_a.dtype();
        auto sbd = scales_b.dtype();
        TVM_FFI_ICHECK(sad.code == kDLFloat && sad.bits == 32)
            << "moe_gemm_fc1_fused: scales_a must be float32";
        TVM_FFI_ICHECK(sbd.code == kDLFloat && sbd.bits == 32)
            << "moe_gemm_fc1_fused: scales_b must be float32";
        TVM_FFI_ICHECK(scales_b.numel() >= static_cast<int64_t>(num_problems) *
                                               (shape_n_interleaved / 128) * (shape_k / 128))
            << "moe_gemm_fc1_fused: scales_b too small for (G, 2I/128, K/128)";

        CHECK_INPUT(d_fp8);
        CHECK_INPUT(sfa2);
        CHECK_DIM(2, d_fp8);
        auto dd = d_fp8.dtype();
        TVM_FFI_ICHECK((dd.code == kDLUInt && dd.bits == 8) || is_fp8_e4m3fn(dd))
            << "moe_gemm_fc1_fused: d_fp8 must be uint8 or float8_e4m3fn";
        TVM_FFI_ICHECK(d_fp8.size(1) == i_size)
            << "moe_gemm_fc1_fused: d_fp8 must be (Mcap, I=" << i_size
            << "), got d_fp8.size(1)=" << d_fp8.size(1);
        auto sd = sfa2.dtype();
        TVM_FFI_ICHECK(sd.code == kDLFloat && sd.bits == 32)
            << "moe_gemm_fc1_fused: sfa2 must be float32";

        CHECK_DEVICE(offsets, a);
        CHECK_DEVICE(b, a);
        CHECK_DEVICE(scales_a, a);
        CHECK_DEVICE(scales_b, a);
        CHECK_DEVICE(d_fp8, a);
        CHECK_DEVICE(sfa2, a);

        auto* runner = selectRunner(/*input_is_fp8=*/true, /*weight_is_fp8=*/true);
        TVM_FFI_ICHECK(runner != nullptr) << "moe_gemm_fc1_fused: no fp8xfp8 runner";
        checkMoeContract("moe_gemm_fc1_fused", runner, num_problems, shape_n_interleaved, shape_k,
                         a);

        int64_t p_stride = runner->getMoePaddedStride();
        TVM_FFI_ICHECK(p_stride > 0)
            << "moe_gemm_fc1_fused: runner has no padded-stride state; call "
               "get_moe_workspace_size first (_Sm90PushMoERunner.configure_workspace does)";
        TVM_FFI_ICHECK(sfa2.numel() >= (i_size / 128) * p_stride)
            << "moe_gemm_fc1_fused: sfa2 has " << sfa2.numel()
            << " elements, needs >= (I/128=" << (i_size / 128) << ") * (P=" << p_stride
            << ") = " << (i_size / 128) * p_stride;
        TVM_FFI_ICHECK(scales_a.numel() >= (shape_k / 128) * p_stride)
            << "moe_gemm_fc1_fused: scales_a has " << scales_a.numel()
            << " elements, needs >= (K/128=" << (shape_k / 128) << ") * (P=" << p_stride
            << ") = " << (shape_k / 128) * p_stride;
        int64_t a_rows_decl = runner->getAlignedARows();
        TVM_FFI_ICHECK(a_rows_decl > 0)
            << "moe_gemm_fc1_fused: runner has no frozen A-row state; call "
               "get_moe_workspace_size first";
        TVM_FFI_ICHECK(a.size(0) >= a_rows_decl)
            << "moe_gemm_fc1_fused: a has " << a.size(0) << " rows, needs >= the frozen TMA "
            << "declaration " << a_rows_decl << " (allocate A at the workspace capacity)";
        TVM_FFI_ICHECK(d_fp8.size(0) >= a.size(0))
            << "moe_gemm_fc1_fused: d_fp8 has " << d_fp8.size(0) << " rows, needs >= a's "
            << a.size(0) << " (valid rows are a prefix of A's row capacity)";

        (void)trusted_offsets;
        auto stream = get_stream(a.device());
        runner->moeGemmFc1Fused(d_fp8.data_ptr(), d_fp8.size(0), std::min(a.size(0), a_rows_decl),
                                static_cast<float*>(sfa2.data_ptr()), a.data_ptr(), b.data_ptr(),
                                static_cast<int64_t const*>(offsets.data_ptr()), num_problems,
                                static_cast<size_t>(shape_n_interleaved),
                                static_cast<size_t>(shape_k), stream,
                                reinterpret_cast<float const*>(scales_a.data_ptr()),
                                reinterpret_cast<float const*>(scales_b.data_ptr()));
      });
    } else if (name == "get_moe_workspace_size") {
      return Function::FromTyped([this](int64_t shape_m, int64_t shape_n, int64_t shape_k,
                                        int64_t top_k, int64_t num_problems, bool a_is_fp8,
                                        bool b_is_fp8) -> int64_t {
        return getMoeWorkspaceSize(shape_m, shape_n, shape_k, top_k, num_problems, a_is_fp8,
                                   b_is_fp8);
      });
    }
    return Function(nullptr);
  }

 private:
  kernels::CutlassFp8BlockScaleGemmRunnerInterface* selectRunner(bool input_is_fp8,
                                                                 bool weight_is_fp8) {
    if (!input_is_fp8 && !weight_is_fp8) {
      return runner_bf16_bf16_.get();
    } else if (!input_is_fp8 && weight_is_fp8) {
      return runner_bf16_fp8_.get();
    } else if (input_is_fp8 && weight_is_fp8) {
      return runner_fp8_fp8_.get();  // W8A8
    } else {
      // FP8 input + BF16 weight is not supported by TensorRT-LLM
      return nullptr;
    }
  }

  void checkWorkspace(const char* what, kernels::CutlassFp8BlockScaleGemmRunnerInterface* runner,
                      const TensorView& input) {
    TVM_FFI_ICHECK(workspace_configured_)
        << what << ": workspace is not configured after the latest size query";
    TVM_FFI_ICHECK(workspace_device_.device_type == input.device().device_type &&
                   workspace_device_.device_id == input.device().device_id)
        << what << ": workspace and input must be on the same CUDA device";
    TVM_FFI_ICHECK(workspace_bytes_ >= runner->getRequiredWorkspaceBytes())
        << what << ": configured workspace has " << workspace_bytes_
        << " bytes, needs >= " << runner->getRequiredWorkspaceBytes();
  }

  void checkMoeContract(const char* what, kernels::CutlassFp8BlockScaleGemmRunnerInterface* runner,
                        size_t num_problems, int64_t shape_n, int64_t shape_k,
                        const TensorView& input) {
    TVM_FFI_ICHECK(moe_workspace_query_done_ && runner == moe_runner_)
        << what << ": call get_moe_workspace_size for this dtype combination first";
    TVM_FFI_ICHECK(runner->getFrozenNumProblems() > 0 && runner->getFrozenMaxShapeN() > 0 &&
                   runner->getFrozenMaxShapeK() > 0)
        << what << ": call get_moe_workspace_size first";
    TVM_FFI_ICHECK_EQ(static_cast<int64_t>(num_problems), runner->getFrozenNumProblems())
        << what << ": runtime group count must match the workspace query";
    TVM_FFI_ICHECK_LE(shape_n, runner->getFrozenMaxShapeN())
        << what << ": runtime N exceeds the workspace query";
    TVM_FFI_ICHECK_LE(shape_k, runner->getFrozenMaxShapeK())
        << what << ": runtime K exceeds the workspace query";
    checkWorkspace(what, runner, input);
  }

  int64_t getMoeWorkspaceSize(int64_t shape_m, int64_t shape_n, int64_t shape_k, int64_t top_k,
                              int64_t num_problems, bool a_is_fp8, bool b_is_fp8) {
    TVM_FFI_ICHECK(shape_m > 0 && shape_n > 0 && shape_k > 0 && top_k > 0 && num_problems > 0)
        << "get_moe_workspace_size: M, N, K, top_k, and num_problems must all be positive";
    TVM_FFI_ICHECK_LE(shape_m, std::numeric_limits<int64_t>::max() / top_k)
        << "get_moe_workspace_size: M * top_k overflows int64";
    TVM_FFI_ICHECK_LE(num_problems, std::numeric_limits<int>::max())
        << "get_moe_workspace_size: num_problems exceeds int32";
    auto* runner = selectRunner(a_is_fp8, b_is_fp8);
    TVM_FFI_ICHECK(runner != nullptr) << "get_moe_workspace_size: unsupported dtype combo";
    auto size = runner->getWorkspaceSize(static_cast<size_t>(shape_m), static_cast<size_t>(shape_n),
                                         static_cast<size_t>(shape_k), static_cast<size_t>(top_k),
                                         static_cast<size_t>(num_problems));
    TVM_FFI_ICHECK_LE(size, static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        << "get_moe_workspace_size: workspace size overflows int64";
    workspace_query_done_ = true;
    workspace_configured_ = false;
    workspace_ = nullptr;
    required_workspace_bytes_ = static_cast<int64_t>(size);
    moe_workspace_query_done_ = true;
    moe_runner_ = runner;
    return required_workspace_bytes_;
  }

  void runGemm(const TensorView& input, const TensorView& weight, const TensorView& output,
               const Optional<TensorView>& scales_a, const Optional<TensorView>& scales_b) {
    auto stream = get_stream(input.device());

    auto input_ptr = input.data_ptr();
    auto weight_ptr = weight.data_ptr();
    auto output_ptr = output.data_ptr();

    int64_t shape_m = input.size(0);
    int64_t shape_k = input.size(1);
    int64_t shape_n = weight.size(0);

    TVM_FFI_ICHECK(input_ptr != nullptr) << "input is null";
    TVM_FFI_ICHECK(weight_ptr != nullptr) << "weight is null";
    TVM_FFI_ICHECK(output_ptr != nullptr) << "output is null";
    TVM_FFI_ICHECK(shape_k == weight.size(1)) << "K dimension mismatch";
    TVM_FFI_ICHECK(shape_k % 16 == 0) << "N must be a multiple of 16, (K=" << shape_k << ")";
    TVM_FFI_ICHECK(shape_n % 16 == 0) << "N must be a multiple of 16, (N=" << shape_n << ")";

    // Determine dtypes for runner selection
    bool input_is_fp8 = is_fp8_e4m3fn(input.dtype());
    bool weight_is_fp8 = is_fp8_e4m3fn(weight.dtype());

    // Validate scale requirements
    if (input_is_fp8) {
      TVM_FFI_ICHECK(scales_a.has_value() && scales_a.value().data_ptr() != nullptr)
          << "scales_a is required for FP8 input";
    }

    if (weight_is_fp8) {
      TVM_FFI_ICHECK(scales_b.has_value() && scales_b.value().data_ptr() != nullptr)
          << "scales_b is required for FP8 weight";
      // Validate scale shape: should be (N, K/128) for per-token or (N/128, K/128) for per-block
      int64_t expected_scale_k = (shape_k + 127) / 128;
      int64_t scale_dim0 = scales_b.value().size(0);
      int64_t scale_dim1 = scales_b.value().size(1);

      bool is_per_token = (scale_dim0 == shape_n && scale_dim1 == expected_scale_k);
      bool is_per_block = (scale_dim0 == (shape_n + 127) / 128 && scale_dim1 == expected_scale_k);

      TVM_FFI_ICHECK(is_per_token || is_per_block)
          << "scales_b shape mismatch: expected (" << shape_n << ", " << expected_scale_k
          << ") for per-token or (" << ((shape_n + 127) / 128) << ", " << expected_scale_k
          << ") for per-block, got (" << scale_dim0 << ", " << scale_dim1 << ")";
    }

    // Extract scale pointers
    float const* scales_a_ptr = scales_a.has_value()
                                    ? reinterpret_cast<float const*>(scales_a.value().data_ptr())
                                    : nullptr;
    float const* scales_b_ptr = scales_b.has_value()
                                    ? reinterpret_cast<float const*>(scales_b.value().data_ptr())
                                    : nullptr;

    // Select appropriate runner
    auto* runner = selectRunner(input_is_fp8, weight_is_fp8);
    TVM_FFI_ICHECK(runner != nullptr) << "Unsupported dtype combination";
    checkWorkspace("run_gemm", runner, input);

    if (input_is_fp8 && weight_is_fp8) {
      // W8A8: Use the pre-quantized FP8 path
      auto* fp8_input = reinterpret_cast<__nv_fp8_e4m3*>(input_ptr);
      auto* fp8_weight = reinterpret_cast<__nv_fp8_e4m3*>(weight_ptr);
      auto* bf16_output = reinterpret_cast<__nv_bfloat16*>(output_ptr);

      runner->gemm(fp8_input, shape_k,    // input with leading dimension
                   fp8_weight, shape_k,   // weight with leading dimension
                   bf16_output, shape_n,  // output with leading dimension
                   shape_m, shape_n, shape_k, scales_a_ptr, scales_b_ptr, stream);
    } else {
      // BF16+BF16 or BF16+FP8: Use internal quantization path
      runner->gemm(output_ptr, input_ptr, weight_ptr, shape_m, shape_n, shape_k, stream,
                   scales_a_ptr, scales_b_ptr);
    }
  }

  void fp8_quantize_1x128(const TensorView input, TensorView valueE4M3, TensorView scaleFP8SF,
                          bool use_ue8m0) {
    auto data_shape = input.sizes();
    TVM_FFI_ICHECK_EQ(data_shape.size(), 2) << "input should be 2D tensor.";

    auto const m = data_shape[0];
    auto const n = data_shape[1];

    TVM_FFI_ICHECK_LE(m, std::numeric_limits<int32_t>::max()) << "M must be within int32";
    TVM_FFI_ICHECK_LE(n, std::numeric_limits<int32_t>::max()) << "N must be within int32";
    TVM_FFI_ICHECK_EQ(n % 16, 0) << "n must be divisible by 16";

    __nv_fp8_e4m3* act_buffer = reinterpret_cast<__nv_fp8_e4m3*>(valueE4M3.data_ptr());
    float* act_scale_buffer = reinterpret_cast<float*>(scaleFP8SF.data_ptr());

    auto stream = get_stream(input.device());

    runner_bf16_fp8_->fp8CS1x128(act_buffer, act_scale_buffer,
                                 reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()), n, m,
                                 stream);
  }

  int64_t getWorkspaceSize(int64_t shape_m, int64_t shape_n, int64_t shape_k) {
    TVM_FFI_ICHECK(shape_m > 0 && shape_n > 0 && shape_k > 0)
        << "get_workspace_size: M, N, and K must all be positive";
    size_t max_size = 0;

    max_size = std::max(max_size, runner_bf16_bf16_->getWorkspaceSize(
                                      static_cast<size_t>(shape_m), static_cast<size_t>(shape_n),
                                      static_cast<size_t>(shape_k), 1, 1));
    max_size = std::max(max_size, runner_bf16_fp8_->getWorkspaceSize(
                                      static_cast<size_t>(shape_m), static_cast<size_t>(shape_n),
                                      static_cast<size_t>(shape_k), 1, 1));
    max_size = std::max(max_size, runner_fp8_fp8_->getWorkspaceSize(
                                      static_cast<size_t>(shape_m), static_cast<size_t>(shape_n),
                                      static_cast<size_t>(shape_k), 1, 1));

    TVM_FFI_ICHECK_LE(max_size, static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        << "get_workspace_size: workspace size overflows int64";
    workspace_query_done_ = true;
    workspace_configured_ = false;
    workspace_ = nullptr;
    required_workspace_bytes_ = static_cast<int64_t>(max_size);
    moe_workspace_query_done_ = false;
    moe_runner_ = nullptr;
    return required_workspace_bytes_;
  }

  void configureWorkspace(const TensorView& workspace) {
    TVM_FFI_ICHECK(workspace_query_done_)
        << "configure_workspace: call a workspace-size query first";
    CHECK_INPUT(workspace);
    auto dtype = workspace.dtype();
    TVM_FFI_ICHECK(dtype.code == kDLUInt && dtype.bits == 8 && dtype.lanes == 1)
        << "configure_workspace: workspace must be uint8";
    TVM_FFI_ICHECK_GE(workspace.numel(), required_workspace_bytes_)
        << "configure_workspace: workspace has " << workspace.numel()
        << " bytes, needs >= " << required_workspace_bytes_;
    auto workspace_ptr = reinterpret_cast<char*>(workspace.data_ptr());
    workspace_ = workspace_ptr;
    workspace_bytes_ = workspace.numel();
    workspace_device_ = workspace.device();
    workspace_configured_ = true;

    runner_bf16_bf16_->configureWorkspace(workspace_ptr);
    runner_bf16_fp8_->configureWorkspace(workspace_ptr);
    runner_fp8_fp8_->configureWorkspace(workspace_ptr);
  }

  std::unique_ptr<
      kernels::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>>
      runner_bf16_bf16_;
  std::unique_ptr<
      kernels::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>
      runner_bf16_fp8_;
  std::unique_ptr<
      kernels::CutlassFp8BlockScaleGemmRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>>
      runner_fp8_fp8_;

  char* workspace_ = nullptr;
  bool workspace_query_done_ = false;
  bool workspace_configured_ = false;
  bool moe_workspace_query_done_ = false;
  kernels::CutlassFp8BlockScaleGemmRunnerInterface* moe_runner_ = nullptr;
  int64_t workspace_bytes_ = 0;
  int64_t required_workspace_bytes_ = 0;
  DLDevice workspace_device_{kDLCPU, 0};
};

tvm::ffi::Module init() {
  auto ptr = tvm::ffi::make_object<Fp8BlockScaleGemmRunner>();
  return tvm::ffi::Module(ptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, init);
