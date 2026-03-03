/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file condition_4d.cu
 * @brief CUDA kernel for conditioning 4D Gaussians onto a 3D time slice.
 *
 * Ports the core math from OMG4's gaussian_model.py:
 *   build_covariance_from_scaling_rotation_4d() and
 *   build_scaling_rotation_4d()
 *
 * Reference: https://arxiv.org/html/2510.03857v1
 */

#include "condition_4d.hpp"
#include "core/logger.hpp"
#include "core/tensor.hpp"

#include <cuda_runtime.h>
#include <stdexcept>

namespace lfs::rendering {

    namespace {

        // -----------------------------------------------------------------------
        // Helper: build a left-isoclinic 4×4 matrix from quaternion (w, x, y, z)
        // -----------------------------------------------------------------------
        __device__ inline void make_left_isoclinic(const float q[4], float L[4][4]) {
            const float w = q[0], x = q[1], y = q[2], z = q[3];
            // Left-isoclinic representation of unit quaternion q
            // Column-major storage (matches the Python @ operator row convention):
            L[0][0] =  w;  L[0][1] = -x;  L[0][2] = -y;  L[0][3] = -z;
            L[1][0] =  x;  L[1][1] =  w;  L[1][2] = -z;  L[1][3] =  y;
            L[2][0] =  y;  L[2][1] =  z;  L[2][2] =  w;  L[2][3] = -x;
            L[3][0] =  z;  L[3][1] = -y;  L[3][2] =  x;  L[3][3] =  w;
        }

        // -----------------------------------------------------------------------
        // Helper: build a right-isoclinic 4×4 matrix from quaternion (w, x, y, z)
        // -----------------------------------------------------------------------
        __device__ inline void make_right_isoclinic(const float q[4], float R[4][4]) {
            const float w = q[0], x = q[1], y = q[2], z = q[3];
            // Right-isoclinic representation
            R[0][0] =  w;  R[0][1] = -x;  R[0][2] = -y;  R[0][3] = -z;
            R[1][0] =  x;  R[1][1] =  w;  R[1][2] =  z;  R[1][3] = -y;
            R[2][0] =  y;  R[2][1] = -z;  R[2][2] =  w;  R[2][3] =  x;
            R[3][0] =  z;  R[3][1] =  y;  R[3][2] = -x;  R[3][3] =  w;
        }

        // -----------------------------------------------------------------------
        // Matrix multiply: C = A @ B  (4×4)
        // -----------------------------------------------------------------------
        __device__ inline void mat4_mul(const float A[4][4], const float B[4][4], float C[4][4]) {
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < 4; ++k)
                        sum += A[i][k] * B[k][j];
                    C[i][j] = sum;
                }
            }
        }

        // -----------------------------------------------------------------------
        // Normalize a quaternion stored as float[4]
        // -----------------------------------------------------------------------
        __device__ inline void normalize_quat(float q[4]) {
            float norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
            if (norm < 1e-8f) norm = 1e-8f;
            for (int i = 0; i < 4; ++i) q[i] /= norm;
        }

        // -----------------------------------------------------------------------
        // Main per-Gaussian conditioning kernel
        // -----------------------------------------------------------------------
        __global__ void condition_4d_gaussians_kernel(
            const float* __restrict__ means_3d,       // [N, 3]
            const float* __restrict__ t_centers,      // [N, 1]
            const float* __restrict__ scaling_xyzt,   // [N, 4]  activated
            const float* __restrict__ rotation_l_in,  // [N, 4]  (w,x,y,z)
            const float* __restrict__ rotation_r_in,  // [N, 4]  (w,x,y,z)
            const float* __restrict__ opacities,      // [N]
            const int N,
            const float playhead_time,
            const float scaling_modifier,
            float* __restrict__ out_means,            // [N, 3]
            float* __restrict__ out_opacity,          // [N]
            float* __restrict__ out_cov6              // [N, 6]  upper-triangle
        ) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= N) return;

            // ---- Load parameters ----
            const float mx = means_3d[i * 3 + 0];
            const float my = means_3d[i * 3 + 1];
            const float mz = means_3d[i * 3 + 2];

            const float t_i = t_centers[i];

            float sl[4], sr[4];
            sl[0] = scaling_xyzt[i * 4 + 0] * scaling_modifier; // sx
            sl[1] = scaling_xyzt[i * 4 + 1] * scaling_modifier; // sy
            sl[2] = scaling_xyzt[i * 4 + 2] * scaling_modifier; // sz
            sl[3] = scaling_xyzt[i * 4 + 3] * scaling_modifier; // st

            float ql[4], qr[4];
            ql[0] = rotation_l_in[i * 4 + 0]; // w
            ql[1] = rotation_l_in[i * 4 + 1]; // x
            ql[2] = rotation_l_in[i * 4 + 2]; // y
            ql[3] = rotation_l_in[i * 4 + 3]; // z
            qr[0] = rotation_r_in[i * 4 + 0];
            qr[1] = rotation_r_in[i * 4 + 1];
            qr[2] = rotation_r_in[i * 4 + 2];
            qr[3] = rotation_r_in[i * 4 + 3];

            normalize_quat(ql);
            normalize_quat(qr);

            // ---- Build L = rotation_l @ rotation_r @ diag(s)  (4×4) ----
            float Rl[4][4], Rr[4][4], RlRr[4][4], L[4][4];
            make_left_isoclinic(ql, Rl);
            make_right_isoclinic(qr, Rr);
            mat4_mul(Rl, Rr, RlRr);

            // Multiply by diagonal scale: L[:,j] = RlRr[:,j] * s[j]
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    L[row][col] = RlRr[row][col] * sl[col];
                }
            }

            // ---- Compute Σ = L @ L^T  (4×4 covariance) ----
            // We only need certain sub-blocks:
            //   Σ_11 = Σ[0:3, 0:3]  (3×3 spatial covariance)
            //   Σ_12 = Σ[0:3, 3]    (3×1 spatial-temporal cross covariance)
            //   Σ_t  = Σ[3, 3]      (scalar temporal variance)

            float cov11[3][3] = {};  // Σ_11
            float cov12[3] = {};     // Σ_12  (column vector)
            float cov_t = 0.0f;      // Σ_t = Σ[3,3]

            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    float v = 0.0f;
                    for (int k = 0; k < 4; ++k) v += L[r][k] * L[c][k];
                    cov11[r][c] = v;
                }
                // Cross term
                float v12 = 0.0f;
                for (int k = 0; k < 4; ++k) v12 += L[r][k] * L[3][k];
                cov12[r] = v12;
            }
            for (int k = 0; k < 4; ++k) cov_t += L[3][k] * L[3][k];

            // ---- Condition on time: Schur complement ----
            // Σ_3d = Σ_11 - Σ_12 @ Σ_12^T / Σ_t
            const float inv_cov_t = 1.0f / (cov_t + 1e-8f);

            float cond_cov11[3][3];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    cond_cov11[r][c] = cov11[r][c] - cov12[r] * cov12[c] * inv_cov_t;
                }
            }

            // ---- Spatial mean offset: Δμ = (Σ_12 / Σ_t) * dt ----
            const float dt = playhead_time - t_i;
            const float scale_dt = inv_cov_t * dt;
            const float dmx = cov12[0] * scale_dt;
            const float dmy = cov12[1] * scale_dt;
            const float dmz = cov12[2] * scale_dt;

            // ---- Marginal opacity: α_t = α * exp(-0.5 * dt² / Σ_t) ----
            const float w_t = expf(-0.5f * dt * dt * inv_cov_t);
            const float alpha_in = opacities[i];
            const float alpha_out = alpha_in * w_t;

            // ---- Write outputs ----
            out_means[i * 3 + 0] = mx + dmx;
            out_means[i * 3 + 1] = my + dmy;
            out_means[i * 3 + 2] = mz + dmz;

            out_opacity[i] = alpha_out;

            // Upper-triangle of symmetric conditioned 3D covariance:
            // [c00, c01, c02, c11, c12, c22]
            out_cov6[i * 6 + 0] = cond_cov11[0][0];
            out_cov6[i * 6 + 1] = cond_cov11[0][1];
            out_cov6[i * 6 + 2] = cond_cov11[0][2];
            out_cov6[i * 6 + 3] = cond_cov11[1][1];
            out_cov6[i * 6 + 4] = cond_cov11[1][2];
            out_cov6[i * 6 + 5] = cond_cov11[2][2];
        }

    } // anonymous namespace

    // -----------------------------------------------------------------------
    // Host wrapper
    // -----------------------------------------------------------------------
    Condition4DResult condition_4d_gaussians(
        const lfs::core::Tensor& means_3d,
        const lfs::core::Tensor& t_centers,
        const lfs::core::Tensor& scaling_xyzt,
        const lfs::core::Tensor& rotation_l,
        const lfs::core::Tensor& rotation_r,
        const lfs::core::Tensor& opacities,
        float playhead_time,
        float scaling_modifier) {

        const int N = static_cast<int>(means_3d.shape()[0]);
        if (N == 0) {
            return {};
        }

        const auto sz = static_cast<size_t>(N);

        // Allocate output tensors on CUDA
        auto out_means = lfs::core::Tensor::empty({sz, 3UL}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        auto out_opacity = lfs::core::Tensor::empty({sz}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        auto out_cov6 = lfs::core::Tensor::empty({sz, 6UL}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        constexpr int BLOCK_SIZE = 256;
        const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

        condition_4d_gaussians_kernel<<<grid_size, BLOCK_SIZE>>>(
            means_3d.ptr<float>(),
            t_centers.ptr<float>(),
            scaling_xyzt.ptr<float>(),
            rotation_l.ptr<float>(),
            rotation_r.ptr<float>(),
            opacities.ptr<float>(),
            N,
            playhead_time,
            scaling_modifier,
            out_means.ptr<float>(),
            out_opacity.ptr<float>(),
            out_cov6.ptr<float>());

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            LOG_ERROR("condition_4d_gaussians kernel failed: {}", cudaGetErrorString(err));
        }

        return {std::move(out_means), std::move(out_opacity), std::move(out_cov6)};
    }

} // namespace lfs::rendering
