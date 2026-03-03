/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data_4d.hpp"
#include "core/logger.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"

#include <cstdint>
#include <stdexcept>

namespace lfs::core {

    // ========== MOVE SEMANTICS ==========

    SplatData4D::SplatData4D(SplatData4D&& other) noexcept
        : SplatData(std::move(static_cast<SplatData&>(other))),
          _t(std::move(other._t)),
          _scaling_t(std::move(other._scaling_t)),
          _rotation_r(std::move(other._rotation_r)),
          time_duration_(other.time_duration_),
          rot_4d_(other.rot_4d_) {}

    SplatData4D& SplatData4D::operator=(SplatData4D&& other) noexcept {
        if (this != &other) {
            SplatData::operator=(std::move(static_cast<SplatData&>(other)));
            _t = std::move(other._t);
            _scaling_t = std::move(other._scaling_t);
            _rotation_r = std::move(other._rotation_r);
            time_duration_ = other.time_duration_;
            rot_4d_ = other.rot_4d_;
        }
        return *this;
    }

    // ========== CONSTRUCTOR ==========

    SplatData4D::SplatData4D(int sh_degree,
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
                             std::array<float, 2> time_duration,
                             bool rot_4d)
        : SplatData(sh_degree,
                    std::move(means),
                    std::move(sh0),
                    std::move(shN),
                    std::move(scaling),
                    std::move(rotation),
                    std::move(opacity),
                    scene_scale),
          _t(std::move(t)),
          _scaling_t(std::move(scaling_t)),
          _rotation_r(std::move(rotation_r)),
          time_duration_(time_duration),
          rot_4d_(rot_4d) {}

    // ========== COMPUTED GETTERS ==========

    Tensor SplatData4D::get_t() const {
        return _t;
    }

    Tensor SplatData4D::get_scaling_t() const {
        // Mirror OMG4: activated scale = exp(raw_scale)
        return _scaling_t.exp();
    }

    Tensor SplatData4D::get_rotation_r() const {
        // Mirror OMG4: normalize the quaternion
        return _rotation_r.normalize(-1);
    }

    Tensor SplatData4D::get_cov_t(float scaling_modifier) const {
        // Build the combined 4D scaling vector: [sx, sy, sz, st] * modifier
        // We extract 3D scale and 1D temporal scale, combine, then compute σ_t
        // from the Schur complement of the 4D covariance.
        //
        // From OMG4 build_covariance_from_scaling_rotation_4d():
        //   L = build_scaling_rotation_4d(modifier * scaling, rotation_l, rotation_r)
        //   Σ = L @ L^T
        //   σ_t = Σ[3,3]  (the temporal variance, before conditioning)
        //
        // For the temporal variance alone (σ_t), we only need the (3,3) component,
        // which equals the squared norm of the 4th row of L.
        //
        // L = R_l @ R_r @ diag(s)
        // The 4th row of L depends only on the 4th column of R_l @ R_r and the
        // temporal scale s_t = exp(_scaling_t) * scaling_modifier.
        //
        // Simplified: σ_t = s_t^2  when rotation is identity (conservative estimate).
        // For the full derivation the caller should use condition_4d_gaussians().
        //
        // Here we return the per-Gaussian temporal variance as s_t^2.
        const Tensor st = get_scaling_t() * scaling_modifier; // [N, 1]
        return st * st;                                        // [N, 1]
    }

    Tensor SplatData4D::get_marginal_t(float timestamp, float scaling_modifier) const {
        // Compute per-Gaussian temporal opacity marginal:
        //   w_t = exp(-0.5 * (t_i - timestamp)^2 / sigma_t_i)
        // where t_i = _t[i] and sigma_t_i = get_cov_t()[i].

        const Tensor t_centers = get_t();                      // [N, 1]
        const Tensor sigma_t = get_cov_t(scaling_modifier);   // [N, 1]

        // dt = t_i - timestamp  [N, 1]
        const Tensor dt = t_centers - timestamp;

        // weight = exp(-0.5 * dt^2 / sigma_t)  [N, 1]
        const Tensor weight = (dt * dt * -0.5f / (sigma_t + 1e-8f)).exp();

        // Return squeezed [N] tensor
        return weight.squeeze(-1);
    }

    // ========== SERIALIZATION ==========

    namespace {
        constexpr uint32_t SPLAT_DATA_4D_MAGIC = 0x4C464534; // "LFE4"
        constexpr uint32_t SPLAT_DATA_4D_VERSION = 1;
    } // namespace

    void SplatData4D::serialize(std::ostream& os) const {
        // Write 4D-specific magic and version header
        os.write(reinterpret_cast<const char*>(&SPLAT_DATA_4D_MAGIC), sizeof(SPLAT_DATA_4D_MAGIC));
        os.write(reinterpret_cast<const char*>(&SPLAT_DATA_4D_VERSION), sizeof(SPLAT_DATA_4D_VERSION));

        // Serialize base 3D data
        SplatData::serialize(os);

        // Serialize 4D temporal parameters
        os << _t << _scaling_t << _rotation_r;

        os.write(reinterpret_cast<const char*>(time_duration_.data()), sizeof(time_duration_));
        const uint8_t rot4d_flag = rot_4d_ ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&rot4d_flag), sizeof(rot4d_flag));

        LOG_DEBUG("Serialized SplatData4D: {} Gaussians, time=[{}, {}]",
                  size(), time_duration_[0], time_duration_[1]);
    }

    void SplatData4D::deserialize(std::istream& is) {
        uint32_t magic = 0, version = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != SPLAT_DATA_4D_MAGIC) {
            throw std::runtime_error("Invalid SplatData4D: wrong magic");
        }
        if (version != SPLAT_DATA_4D_VERSION) {
            throw std::runtime_error("Unsupported SplatData4D version: " + std::to_string(version));
        }

        // Deserialize base 3D data
        SplatData::deserialize(is);

        // Deserialize 4D temporal parameters
        Tensor t, scaling_t, rotation_r;
        is >> t >> scaling_t >> rotation_r;

        _t = std::move(t).cuda();
        _scaling_t = std::move(scaling_t).cuda();
        _rotation_r = std::move(rotation_r).cuda();

        is.read(reinterpret_cast<char*>(time_duration_.data()), sizeof(time_duration_));
        uint8_t rot4d_flag = 0;
        is.read(reinterpret_cast<char*>(&rot4d_flag), sizeof(rot4d_flag));
        rot_4d_ = (rot4d_flag != 0);

        LOG_DEBUG("Deserialized SplatData4D: {} Gaussians, time=[{}, {}]",
                  size(), time_duration_[0], time_duration_[1]);
    }

} // namespace lfs::core
