/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"

namespace lfs::rendering {

    /**
     * @brief Condition 4D Gaussians onto a 3D slice at a given playhead time.
     *
     * For each 4D Gaussian i, given the 4D parameters:
     *   - means_3d[i]    : 3D spatial center [x, y, z]
     *   - t_centers[i]   : temporal center
     *   - scaling_xyzt[i]: 4D scale [sx, sy, sz, st] (activated, not log)
     *   - rotation_l[i]  : left-isoclinic quaternion [w, x, y, z]
     *   - rotation_r[i]  : right-isoclinic quaternion [w, x, y, z]
     *   - opacities[i]   : pre-activated opacity
     *
     * This function computes the conditioned 3D Gaussian at time `playhead_time`:
     *   - dt = playhead_time - t_centers[i]
     *   - Computes Schur complement of 4D covariance to get conditioned 3D cov
     *   - Outputs conditioned 3D means (original + spatial offset from time)
     *   - Outputs marginal opacity (original * exp(-0.5 * dt² / sigma_t))
     *   - Outputs conditioned 3D covariance as 6 upper-triangle components
     *
     * The output conditioned covariance and means can be fed directly into the
     * existing gsplat rasterization pipeline as a standard 3DGS model.
     *
     * @param means_3d    [N, 3] Gaussian spatial centers (CUDA, float32).
     * @param t_centers   [N, 1] Temporal centers (CUDA, float32).
     * @param scaling_xyzt [N, 4] Activated 4D scales [sx, sy, sz, st] (CUDA, float32).
     * @param rotation_l  [N, 4] Left-isoclinic quaternions [w, x, y, z] (CUDA, float32).
     * @param rotation_r  [N, 4] Right-isoclinic quaternions [w, x, y, z] (CUDA, float32).
     * @param opacities   [N] Pre-activated opacities (CUDA, float32).
     * @param playhead_time  Current time in seconds.
     * @param scaling_modifier Global scale multiplier.
     * @return Tuple of:
     *   - conditioned_means   [N, 3]  (CUDA float32)
     *   - conditioned_opacity [N]     (CUDA float32, marginal-weighted)
     *   - conditioned_cov6    [N, 6]  Upper-triangle of conditioned 3D covariance
     *                                 layout: [c00, c01, c02, c11, c12, c22]
     */
    struct Condition4DResult {
        lfs::core::Tensor conditioned_means;   ///< [N, 3]
        lfs::core::Tensor conditioned_opacity; ///< [N]
        lfs::core::Tensor conditioned_cov6;    ///< [N, 6]  upper-triangle
    };

    Condition4DResult condition_4d_gaussians(
        const lfs::core::Tensor& means_3d,
        const lfs::core::Tensor& t_centers,
        const lfs::core::Tensor& scaling_xyzt,
        const lfs::core::Tensor& rotation_l,
        const lfs::core::Tensor& rotation_r,
        const lfs::core::Tensor& opacities,
        float playhead_time,
        float scaling_modifier = 1.0f);

} // namespace lfs::rendering
