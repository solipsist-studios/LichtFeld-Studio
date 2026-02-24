/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include <algorithm>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace lfs::training {

    /// Per-frame file paths for a single (camera, time) pair.
    struct SequenceFrameEntry {
        std::filesystem::path image_path;
        std::optional<std::filesystem::path> mask_path;
    };

    /**
     * @brief Time-indexed multi-camera dataset for 4D training.
     *
     * Holds a fixed set of cameras whose intrinsics/extrinsics do not change
     * over time, a list of discrete time steps (timestamps in seconds, or
     * frame indices cast to float), and a dense frame table mapping every
     * (time_idx, cam_idx) pair to on-disk paths.
     *
     * Access patterns:
     *   - All cameras at a given time step: get_time_slice(time_idx)
     *   - Single (camera, time) frame:      get_frame(cam_idx, time_idx)
     *   - Nearest time step for playhead:   get_time_step_for_time(seconds)
     */
    class SequenceDataset {
    public:
        /**
         * @param cameras    Fixed cameras (intrinsics/extrinsics constant over time).
         * @param timestamps Discrete time steps in seconds (monotonically increasing).
         * @param frames     frames[time_idx][cam_idx] = SequenceFrameEntry.
         *                   Must satisfy frames.size() == timestamps.size() and
         *                   frames[t].size() == cameras.size() for all t.
         */
        SequenceDataset(
            std::vector<std::shared_ptr<lfs::core::Camera>> cameras,
            std::vector<float> timestamps,
            std::vector<std::vector<SequenceFrameEntry>> frames)
            : cameras_(std::move(cameras)),
              timestamps_(std::move(timestamps)),
              frames_(std::move(frames)) {

            if (frames_.size() != timestamps_.size()) {
                throw std::invalid_argument(
                    std::format("SequenceDataset: frames.size() ({}) != timestamps.size() ({})",
                                frames_.size(), timestamps_.size()));
            }
            for (size_t t = 0; t < frames_.size(); ++t) {
                if (frames_[t].size() != cameras_.size()) {
                    throw std::invalid_argument(
                        std::format("SequenceDataset: frames[{}].size() ({}) != cameras.size() ({})",
                                    t, frames_[t].size(), cameras_.size()));
                }
            }
        }

        /// Number of cameras (fixed across all time steps).
        size_t num_cameras() const noexcept { return cameras_.size(); }

        /// Number of discrete time steps.
        size_t num_timesteps() const noexcept { return timestamps_.size(); }

        /// Timestamp (seconds) for the given time index.
        float get_timestamp(size_t time_idx) const {
            if (time_idx >= timestamps_.size())
                throw std::out_of_range("time_idx out of range");
            return timestamps_[time_idx];
        }

        /// All cameras (shared ownership, fixed across time).
        const std::vector<std::shared_ptr<lfs::core::Camera>>& get_cameras() const noexcept {
            return cameras_;
        }

        /// All camera frames for the given time step (one entry per camera).
        const std::vector<SequenceFrameEntry>& get_time_slice(size_t time_idx) const {
            if (time_idx >= frames_.size())
                throw std::out_of_range("time_idx out of range");
            return frames_[time_idx];
        }

        /// Single frame for (camera_idx, time_idx).
        const SequenceFrameEntry& get_frame(size_t cam_idx, size_t time_idx) const {
            if (time_idx >= frames_.size())
                throw std::out_of_range("time_idx out of range");
            if (cam_idx >= frames_[time_idx].size())
                throw std::out_of_range("cam_idx out of range");
            return frames_[time_idx][cam_idx];
        }

        /**
         * @brief Return the index of the time step nearest to @p time_seconds.
         *
         * Uses binary search on the sorted timestamps vector.  Returns 0 if the
         * dataset has no time steps.
         */
        size_t get_time_step_for_time(float time_seconds) const noexcept {
            if (timestamps_.empty())
                return 0;

            // Lower-bound search then pick nearest neighbour.
            auto it = std::lower_bound(timestamps_.begin(), timestamps_.end(), time_seconds);
            if (it == timestamps_.end())
                return timestamps_.size() - 1;
            if (it == timestamps_.begin())
                return 0;

            const size_t upper = static_cast<size_t>(it - timestamps_.begin());
            const size_t lower = upper - 1;
            const float d_lower = time_seconds - timestamps_[lower];
            const float d_upper = timestamps_[upper] - time_seconds;
            return d_lower <= d_upper ? lower : upper;
        }

    private:
        std::vector<std::shared_ptr<lfs::core::Camera>> cameras_;
        std::vector<float> timestamps_;
        std::vector<std::vector<SequenceFrameEntry>> frames_; // [time_idx][cam_idx]
    };

} // namespace lfs::training
