/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

// non-persistent-cooperative GEMM
namespace tensorrt_llm::kernels::fp8_blockscale_gemm {

class CutlassFp8BlockScaleGemmRunnerInterface {
 public:
  CutlassFp8BlockScaleGemmRunnerInterface() {}

  virtual ~CutlassFp8BlockScaleGemmRunnerInterface() {}

  virtual void gemm(void* mat_d, void const* mat_a, void const* mat_b, int shape_m, int shape_n,
                    int shape_k, cudaStream_t stream, float const* scales_a = nullptr,
                    float const* scales_b = nullptr) = 0;

  virtual void gemm(__nv_fp8_e4m3 const* mat_a, int ld_a, __nv_fp8_e4m3 const* mat_b, int ld_b,
                    __nv_bfloat16* mat_d, int ld_d, int shape_m, int shape_n, int shape_k,
                    float const* scales_a, float const* scales_b, cudaStream_t stream) = 0;

  virtual void moeGemm(void* mat_d, void const* mat_a, void const* mat_b,
                       int64_t const* problem_m_offsets, size_t num_problems, size_t shape_n,
                       size_t shape_k, cudaStream_t stream, float const* scales_a = nullptr,
                       float const* scales_b = nullptr) = 0;

  virtual void moeGemmFc1Fused(void* mat_d_fp8, int64_t d_rows, int64_t a_rows, float* sfa_out,
                               void const* mat_a, void const* mat_b,
                               int64_t const* problem_m_offsets, size_t num_problems,
                               size_t shape_n_interleaved, size_t shape_k, cudaStream_t stream,
                               float const* scales_a, float const* scales_b) = 0;

  // Workspace-frozen padded-M stride P of the grouped sfa layouts (0 until
  // getWorkspaceSize); bindings size-check caller buffers against it.
  virtual int64_t getMoePaddedStride() const = 0;

  // The workspace-frozen row count the grouped A-side TMA descriptors are
  // DECLARED with (max_shape_m_4_align_; 0 until getWorkspaceSize has run).
  // TMA clamps loads to the DECLARED box, not the allocation, so an fp8 A
  // buffer with fewer rows than this is readable out of bounds through the
  // descriptor even when every offset is valid -- bindings must size-check
  // the caller's A (and its grouped scales) against it.
  virtual int64_t getAlignedARows() const = 0;
  virtual int64_t getFrozenNumProblems() const = 0;
  virtual int64_t getFrozenMaxShapeN() const = 0;
  virtual int64_t getFrozenMaxShapeK() const = 0;
  virtual int64_t getRequiredWorkspaceBytes() const = 0;

  virtual void strideBatchGemm(__nv_bfloat16* mat_d, int ld_d, int stride_d, __nv_fp8_e4m3* mat_a,
                               int ld_a, int stride_a, __nv_fp8_e4m3* mat_b, int ld_b, int stride_b,
                               int num_problems, int shape_m, int shape_n, int shape_k,
                               cudaStream_t stream, float* scales_a, int stride_scales_a,
                               float* scales_b) = 0;

  virtual void fp8CS1x128(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat,
                          int shape_x, int shape_y, cudaStream_t stream) = 0;
  virtual void fp8CS1x128Reshape(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat,
                                 int shape_x, int shape_h, int shape_y, int stride_x,
                                 cudaStream_t stream) = 0;
  virtual void fp8CS128x128(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat,
                            int shape_x, int shape_y, cudaStream_t stream) = 0;
  // Returns desired workspace size in bytes.
  virtual size_t getWorkspaceSizeBase(size_t max_shape_m, size_t shape_n, size_t shape_k,
                                      size_t num_problems = 1) = 0;
  virtual size_t getWorkspaceSize(size_t shape_m, size_t shape_n, size_t shape_k, size_t top_k = 1,
                                  size_t num_problems = 1) = 0;

  void configureWorkspace(char* ws_ptr) { workspace_ = ws_ptr; }

  virtual size_t getFP8DataSize(int shape_m, int shape_n, bool is_act) = 0;
  virtual size_t getActScaleSize(int shape_m, int shape_k) = 0;
  virtual size_t getWeightScaleSize(int shape_n, int shape_k) = 0;
  virtual size_t getActWorkspaceSize(int shape_m, int shape_k) = 0;
  virtual size_t getWeightWorkspaceSize(int shape_n, int shape_k) = 0;

 protected:
  char* workspace_ = nullptr;
};

template <typename ElementA, typename ElementB, typename ElementD>
class CutlassFp8BlockScaleGemmRunner : public CutlassFp8BlockScaleGemmRunnerInterface {
 public:
  CutlassFp8BlockScaleGemmRunner() = default;
  ~CutlassFp8BlockScaleGemmRunner() override = default;

  void gemm(void* mat_d, void const* mat_a, void const* mat_b, int shape_m, int shape_n,
            int shape_k, cudaStream_t stream, float const* scales_a = nullptr,
            float const* scales_b = nullptr) override;

  void gemm(__nv_fp8_e4m3 const* mat_a, int ld_a, __nv_fp8_e4m3 const* mat_b, int ld_b,
            __nv_bfloat16* mat_d, int ld_d, int shape_m, int shape_n, int shape_k,
            float const* scales_a, float const* scales_b, cudaStream_t stream) override;

  void moeGemm(void* mat_d, void const* mat_a, void const* mat_b, int64_t const* problem_m_offsets,
               size_t num_problems, size_t shape_n, size_t shape_k, cudaStream_t stream,
               float const* scales_a = nullptr, float const* scales_b = nullptr) override;

  void moeGemmFc1Fused(void* mat_d_fp8, int64_t d_rows, int64_t a_rows, float* sfa_out,
                       void const* mat_a, void const* mat_b, int64_t const* problem_m_offsets,
                       size_t num_problems, size_t shape_n_interleaved, size_t shape_k,
                       cudaStream_t stream, float const* scales_a, float const* scales_b) override;

  int64_t getMoePaddedStride() const override { return max_shape_m_32_align_padded_; }

  int64_t getAlignedARows() const override { return max_shape_m_4_align_; }
  int64_t getFrozenNumProblems() const override { return frozen_num_problems_; }
  int64_t getFrozenMaxShapeN() const override { return frozen_max_shape_n_; }
  int64_t getFrozenMaxShapeK() const override { return frozen_max_shape_k_; }
  int64_t getRequiredWorkspaceBytes() const override { return required_workspace_bytes_; }

  void strideBatchGemm(__nv_bfloat16* mat_d, int ld_d, int stride_d, __nv_fp8_e4m3* mat_a, int ld_a,
                       int stride_a, __nv_fp8_e4m3* mat_b, int ld_b, int stride_b, int num_problems,
                       int shape_m, int shape_n, int shape_k, cudaStream_t stream, float* scales_a,
                       int stride_scales_a, float* scales_b) override;

  void fp8CS1x128(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x,
                  int shape_y, cudaStream_t stream) override;
  void fp8CS1x128Reshape(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat,
                         int shape_x, int shape_h, int shape_y, int stride_x,
                         cudaStream_t stream) override;
  void fp8CS128x128(__nv_fp8_e4m3* mat_quant, float* scales, __nv_bfloat16 const* mat, int shape_x,
                    int shape_y, cudaStream_t stream) override;

  // Returns desired workspace size in bytes.
  size_t getWorkspaceSizeBase(size_t max_shape_m, size_t shape_n, size_t shape_k,
                              size_t num_problems = 1) override;
  size_t getWorkspaceSize(size_t shape_m, size_t shape_n, size_t shape_k, size_t top_k = 1,
                          size_t num_problems = 1) override;

  size_t getFP8DataSize(int shape_m, int shape_n, bool is_act) override;
  size_t getActScaleSize(int shape_m, int shape_k) override;
  size_t getWeightScaleSize(int shape_n, int shape_k) override;
  size_t getActWorkspaceSize(int shape_m, int shape_k) override;
  size_t getWeightWorkspaceSize(int shape_n, int shape_k) override;

 private:
  int64_t max_shape_m_4_align_ = 0;
  int64_t max_shape_m_32_align_padded_ = 0;
  int64_t expected_m_ = 0;
  int64_t frozen_num_problems_ = 0;
  int64_t frozen_max_shape_n_ = 0;
  int64_t frozen_max_shape_k_ = 0;
  int64_t required_workspace_bytes_ = 0;
};

}  // namespace tensorrt_llm::kernels::fp8_blockscale_gemm
