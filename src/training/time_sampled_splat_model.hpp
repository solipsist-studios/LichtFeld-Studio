/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**
 * @file time_sampled_splat_model.hpp
 * @brief Container for a sequence of per-frame 3D Gaussian Splat models (Milestone 2).
 *
 * TimeSampledSplatModel holds an ordered list of (timestamp, model) entries that
 * represent the output of Flipbook/per-frame training over a 4D dataset.
 * Each entry may carry an in-memory SplatData model and/or an on-disk path for
 * deferred loading.  Nearest-frame selection by playhead time is built in.
 */

#include "core/splat_data.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

namespace lfs::training {

    /**
     * @brief A sequence of trained 3D Gaussian Splat models indexed by discrete time.
     *
     * This is the primary result representation produced by the Flipbook training mode.
     * Entries must be added in monotonically increasing timestamp order.
     *
     * Playback / rendering pattern:
     *   size_t idx = model.get_model_index_for_time(global_time);
     *   const Entry& e = model.get_entry(idx);
     *   // render e.model (or load from e.model_path if e.model is null)
     */
    class TimeSampledSplatModel {
    public:
        /// A single time-step entry: timestamp + optional in-memory model + optional on-disk path.
        struct Entry {
            float timestamp;                              ///< Playhead position in seconds.
            std::filesystem::path model_path;             ///< On-disk path (PLY/SOG/SPZ). May be empty.
            std::shared_ptr<lfs::core::SplatData> model; ///< In-memory model. May be null.
        };

        TimeSampledSplatModel() = default;

        /**
         * @brief Append a time-step entry.
         *
         * @param timestamp  Must be >= the timestamp of the previously added entry.
         * @param model_path On-disk path for the model (may be empty when model is provided).
         * @param model      In-memory SplatData (may be null when model_path is provided).
         * @throws std::invalid_argument if timestamp is less than the last entry's timestamp.
         */
        void add_entry(float timestamp,
                       std::filesystem::path model_path,
                       std::shared_ptr<lfs::core::SplatData> model = nullptr) {
            if (!entries_.empty() && timestamp < entries_.back().timestamp) {
                throw std::invalid_argument(
                    "TimeSampledSplatModel: timestamps must be monotonically increasing");
            }
            entries_.push_back({timestamp, std::move(model_path), std::move(model)});
        }

        /// Number of time-step entries.
        [[nodiscard]] size_t size() const noexcept { return entries_.size(); }

        /// True when no entries have been added.
        [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

        /// All entries in timestamp order.
        [[nodiscard]] const std::vector<Entry>& entries() const noexcept { return entries_; }

        /// Timestamp (seconds) for the entry at @p index.
        /// @throws std::out_of_range
        [[nodiscard]] float get_timestamp(size_t index) const {
            if (index >= entries_.size())
                throw std::out_of_range("TimeSampledSplatModel: index out of range");
            return entries_[index].timestamp;
        }

        /**
         * @brief Return the index of the entry whose timestamp is nearest to @p time_seconds.
         *
         * Uses nearest-neighbour (round-to-nearest) semantics.  If the model has no entries,
         * returns 0 (no-op, callers should check empty() first).
         */
        [[nodiscard]] size_t get_model_index_for_time(float time_seconds) const noexcept {
            if (entries_.empty())
                return 0;

            auto it = std::lower_bound(entries_.begin(), entries_.end(), time_seconds,
                                       [](const Entry& e, float t) { return e.timestamp < t; });

            if (it == entries_.end())
                return entries_.size() - 1;
            if (it == entries_.begin())
                return 0;

            const size_t upper = static_cast<size_t>(it - entries_.begin());
            const size_t lower = upper - 1;
            const float d_lower = time_seconds - entries_[lower].timestamp;
            const float d_upper = entries_[upper].timestamp - time_seconds;
            return d_lower <= d_upper ? lower : upper;
        }

        /// Entry at @p index.
        /// @throws std::out_of_range
        [[nodiscard]] const Entry& get_entry(size_t index) const {
            if (index >= entries_.size())
                throw std::out_of_range("TimeSampledSplatModel: index out of range");
            return entries_[index];
        }

        /// Convenience: entry nearest to @p time_seconds.
        [[nodiscard]] const Entry& get_entry_for_time(float time_seconds) const {
            return entries_[get_model_index_for_time(time_seconds)];
        }

    private:
        std::vector<Entry> entries_; ///< Sorted by timestamp (monotonically increasing).
    };

} // namespace lfs::training
