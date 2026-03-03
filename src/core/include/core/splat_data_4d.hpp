/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"

#include <array>
#include <expected>
#include <iosfwd>
#include <string>

namespace lfs::core {

    /**
     * @brief Extended Gaussian splat data with temporal (4D) attributes for OMG4 models.
     *
     * Inherits all 3D Gaussian attributes from SplatData and adds the temporal
     * parameters required by 4D Gaussian Splatting (OMG4):
     *   - Temporal center  _t            [N, 1]
     *   - Temporal scale   _scaling_t    [N, 1]  (log-space, as in 3D case)
     *   - Right-isoclinic rotation _rotation_r [N, 4]  (4D rotation quaternion)
     *
     * Computed getters mirror OMG4's GaussianModel Python class:
     *   - get_t()            → raw temporal center
     *   - get_scaling_t()    → exp(_scaling_t)  (activated)
     *   - get_rotation_r()   → normalize(_rotation_r)
     *   - get_marginal_t()   → temporal opacity weight exp(-0.5 * (t-τ)²/σ_t)
     *
     * Reference: https://arxiv.org/html/2510.03857v1
     */
    class LFS_CORE_API SplatData4D : public SplatData {
    public:
        SplatData4D() = default;
        ~SplatData4D() = default;

        // Delete copy, allow move
        SplatData4D(const SplatData4D&) = delete;
        SplatData4D& operator=(const SplatData4D&) = delete;
        SplatData4D(SplatData4D&& other) noexcept;
        SplatData4D& operator=(SplatData4D&& other) noexcept;

        /**
         * @brief Construct SplatData4D from all 3D and 4D parameters.
         *
         * @param sh_degree   Spherical harmonics degree.
         * @param means       Gaussian centers [N, 3].
         * @param sh0         DC SH coefficients [N, 3].
         * @param shN         Higher-order SH coefficients [N, K] (may be empty).
         * @param scaling     3D scale (log-space) [N, 3].
         * @param rotation    3D rotation quaternion [N, 4].
         * @param opacity     Raw opacity (logit) [N, 1].
         * @param scene_scale Scene scale factor.
         * @param t           Temporal center [N, 1].
         * @param scaling_t   Temporal scale (log-space) [N, 1].
         * @param rotation_r  Right-isoclinic 4D rotation quaternion [N, 4].
         * @param time_duration Time range [start, end] in seconds.
         * @param rot_4d      Whether 4D rotation is enabled.
         */
        SplatData4D(int sh_degree,
                    Tensor means,
                    Tensor sh0,
                    Tensor shN,
                    Tensor scaling,
                    Tensor rotation,
                    Tensor opacity,
                    float scene_scale,
                    Tensor t,
                    Tensor scaling_t,
                    Tensor rotation_r,
                    std::array<float, 2> time_duration = {0.0f, 1.0f},
                    bool rot_4d = true);

        // ========== Computed getters (mirror OMG4 GaussianModel) ==========

        /// Raw temporal center positions [N, 1].
        Tensor get_t() const;

        /// Activated temporal scale: exp(_scaling_t) [N, 1].
        Tensor get_scaling_t() const;

        /// Normalized right-isoclinic rotation quaternion [N, 4].
        Tensor get_rotation_r() const;

        /**
         * @brief Compute the temporal variance σ_t from the 4×4 covariance.
         *
         * Uses the Schur complement of the full 4D covariance to isolate the
         * temporal component.  scaling_modifier scales all spatial+temporal scales.
         *
         * @param scaling_modifier  Global scale multiplier (default 1.0).
         * @return Tensor of shape [N, 1]: per-Gaussian temporal variance.
         */
        Tensor get_cov_t(float scaling_modifier = 1.0f) const;

        /**
         * @brief Compute the temporal marginal opacity weight.
         *
         * Returns exp(-0.5 * (t - timestamp)² / σ_t) for each Gaussian, where
         * σ_t is the temporal variance from get_cov_t().
         *
         * @param timestamp        Current playhead time (seconds).
         * @param scaling_modifier Global scale multiplier (default 1.0).
         * @return Tensor of shape [N]: per-Gaussian marginal opacity weight ∈ (0, 1].
         */
        Tensor get_marginal_t(float timestamp, float scaling_modifier = 1.0f) const;

        // ========== Raw tensor accessors ==========
        inline Tensor& t_raw() { return _t; }
        inline const Tensor& t_raw() const { return _t; }
        inline Tensor& scaling_t_raw() { return _scaling_t; }
        inline const Tensor& scaling_t_raw() const { return _scaling_t; }
        inline Tensor& rotation_r_raw() { return _rotation_r; }
        inline const Tensor& rotation_r_raw() const { return _rotation_r; }

        // ========== Time range ==========
        /// Time range [start, end] in seconds.
        const std::array<float, 2>& time_duration() const { return time_duration_; }
        void set_time_duration(std::array<float, 2> dur) { time_duration_ = dur; }

        /// Whether 4D rotation (rotation_r) is enabled.
        bool has_rot_4d() const { return rot_4d_; }
        void set_rot_4d(bool enabled) { rot_4d_ = enabled; }

        // ========== Serialization ==========
        void serialize(std::ostream& os) const;
        void deserialize(std::istream& is);

    private:
        // 4D temporal parameters
        Tensor _t;          ///< Temporal center [N, 1]
        Tensor _scaling_t;  ///< Temporal scale log-space [N, 1]
        Tensor _rotation_r; ///< Right-isoclinic 4D rotation quaternion [N, 4]

        std::array<float, 2> time_duration_{0.0f, 1.0f}; ///< [start, end] in seconds
        bool rot_4d_ = true; ///< Whether 4D rotation is used
    };

} // namespace lfs::core
