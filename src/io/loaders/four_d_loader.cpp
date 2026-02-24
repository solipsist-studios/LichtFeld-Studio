/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/loaders/four_d_loader.hpp"
#include "core/camera.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "io/error.hpp"
#include "io/filesystem_utils.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>

namespace lfs::io {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;

    namespace {
        /// Name of the manifest file that identifies a 4D dataset directory.
        constexpr const char* MANIFEST_FILENAME = "dataset4d.json";

        bool has_manifest(const std::filesystem::path& dir) {
            return safe_exists(dir / MANIFEST_FILENAME);
        }

        /// Parse a 3x3 rotation matrix from a JSON array-of-arrays.
        Tensor parse_R(const nlohmann::json& j) {
            // Accepts [[r00,r01,r02],[r10,...],[r20,...]]
            std::vector<float> vals;
            vals.reserve(9);
            for (const auto& row : j) {
                for (float v : row) {
                    vals.push_back(v);
                }
            }
            if (vals.size() != 9) {
                throw std::runtime_error("R must be a 3x3 array");
            }
            return Tensor::from_vector(vals, {3, 3}, Device::CPU);
        }

        /// Parse a translation vector from a JSON array of 3 floats.
        Tensor parse_T(const nlohmann::json& j) {
            std::vector<float> vals;
            vals.reserve(3);
            for (float v : j) {
                vals.push_back(v);
            }
            if (vals.size() != 3) {
                throw std::runtime_error("T must be a 3-element array");
            }
            return Tensor::from_vector(vals, {3}, Device::CPU);
        }
    } // namespace

    // -------------------------------------------------------------------------

    bool FourDLoader::canLoad(const std::filesystem::path& path) const {
        if (!safe_exists(path))
            return false;
        if (safe_is_directory(path))
            return has_manifest(path);
        // Also accept the manifest file itself
        return path.filename() == MANIFEST_FILENAME && safe_exists(path);
    }

    std::string FourDLoader::name() const { return "4D"; }

    std::vector<std::string> FourDLoader::supportedExtensions() const { return {}; }

    int FourDLoader::priority() const { return 10; }

    // -------------------------------------------------------------------------

    Result<LoadResult> FourDLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        const auto start_time = std::chrono::high_resolution_clock::now();

        // Resolve the manifest file path.
        std::filesystem::path manifest_path;
        if (safe_is_directory(path)) {
            manifest_path = path / MANIFEST_FILENAME;
        } else if (path.filename() == MANIFEST_FILENAME) {
            manifest_path = path;
        } else {
            return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                              "Path must be a directory containing dataset4d.json "
                              "or the manifest file itself",
                              path);
        }

        if (!safe_exists(manifest_path)) {
            return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                              "dataset4d.json not found", manifest_path);
        }

        const std::filesystem::path base_dir = manifest_path.parent_path();

        if (options.progress)
            options.progress(0.0f, "Loading 4D dataset...");

        // ------------------------------------------------------------------ //
        // Parse manifest JSON                                                 //
        // ------------------------------------------------------------------ //
        nlohmann::json j;
        {
            std::ifstream f;
            if (!lfs::core::open_file_for_read(manifest_path, f)) {
                return make_error(ErrorCode::PERMISSION_DENIED,
                                  "Cannot open dataset4d.json for reading", manifest_path);
            }
            try {
                f >> j;
            } catch (const std::exception& e) {
                return make_error(ErrorCode::MALFORMED_JSON,
                                  std::format("Invalid JSON: {}", e.what()), manifest_path);
            }
        }

        // Validate required top-level fields.
        if (!j.contains("cameras") || !j["cameras"].is_array()) {
            return make_error(ErrorCode::INVALID_DATASET,
                              "dataset4d.json: missing 'cameras' array", manifest_path);
        }
        if (!j.contains("frames") || !j["frames"].is_array()) {
            return make_error(ErrorCode::INVALID_DATASET,
                              "dataset4d.json: missing 'frames' array", manifest_path);
        }

        // ------------------------------------------------------------------ //
        // Parse timestamps                                                    //
        // ------------------------------------------------------------------ //
        std::vector<float> timestamps;
        if (j.contains("timestamps") && j["timestamps"].is_array()) {
            for (float t : j["timestamps"]) {
                timestamps.push_back(t);
            }
            // Validate monotonicity.
            for (size_t i = 1; i < timestamps.size(); ++i) {
                if (timestamps[i] <= timestamps[i - 1]) {
                    return make_error(ErrorCode::INVALID_DATASET,
                                      std::format("timestamps must be strictly increasing "
                                                  "(index {} value {:.6f} <= index {} value {:.6f})",
                                                  i, timestamps[i], i - 1, timestamps[i - 1]),
                                      manifest_path);
                }
            }
        }

        if (options.progress)
            options.progress(10.0f, "Parsing cameras...");

        // ------------------------------------------------------------------ //
        // Parse cameras                                                       //
        // ------------------------------------------------------------------ //
        std::unordered_map<std::string, size_t> cam_id_to_idx;
        std::vector<std::shared_ptr<lfs::core::Camera>> cameras;

        const auto& jcams = j["cameras"];
        cameras.reserve(jcams.size());

        for (size_t ci = 0; ci < jcams.size(); ++ci) {
            const auto& jc = jcams[ci];

            auto require_field = [&](const char* field) -> Result<void> {
                if (!jc.contains(field)) {
                    return make_error(ErrorCode::INVALID_DATASET,
                                      std::format("camera[{}]: missing field '{}'", ci, field),
                                      manifest_path);
                }
                return {};
            };

            if (auto r = require_field("id"); !r)
                return std::unexpected(r.error());
            if (auto r = require_field("width"); !r)
                return std::unexpected(r.error());
            if (auto r = require_field("height"); !r)
                return std::unexpected(r.error());
            if (auto r = require_field("focal_x"); !r)
                return std::unexpected(r.error());
            if (auto r = require_field("focal_y"); !r)
                return std::unexpected(r.error());
            if (auto r = require_field("center_x"); !r)
                return std::unexpected(r.error());
            if (auto r = require_field("center_y"); !r)
                return std::unexpected(r.error());
            if (auto r = require_field("R"); !r)
                return std::unexpected(r.error());
            if (auto r = require_field("T"); !r)
                return std::unexpected(r.error());

            const std::string cam_id = jc["id"].get<std::string>();
            if (cam_id_to_idx.count(cam_id)) {
                return make_error(ErrorCode::INVALID_DATASET,
                                  std::format("Duplicate camera id '{}'", cam_id), manifest_path);
            }

            Tensor R_mat, T_vec;
            try {
                R_mat = parse_R(jc["R"]);
                T_vec = parse_T(jc["T"]);
            } catch (const std::exception& e) {
                return make_error(ErrorCode::INVALID_DATASET,
                                  std::format("camera '{}': {}", cam_id, e.what()), manifest_path);
            }

            const int width = jc["width"].get<int>();
            const int height = jc["height"].get<int>();
            const float focal_x = jc["focal_x"].get<float>();
            const float focal_y = jc["focal_y"].get<float>();
            const float center_x = jc["center_x"].get<float>();
            const float center_y = jc["center_y"].get<float>();

            Tensor empty_dist;

            auto cam = std::make_shared<lfs::core::Camera>(
                R_mat,
                T_vec,
                focal_x,
                focal_y,
                center_x,
                center_y,
                empty_dist, // radial distortion (none)
                empty_dist, // tangential distortion (none)
                lfs::core::CameraModelType::PINHOLE,
                cam_id,                  // image_name = camera id (no per-frame path at this level)
                std::filesystem::path{}, // image_path filled per-frame
                std::filesystem::path{}, // mask_path filled per-frame
                width,
                height,
                static_cast<int>(ci));

            cam_id_to_idx[cam_id] = ci;
            cameras.push_back(std::move(cam));
        }

        if (cameras.empty()) {
            return make_error(ErrorCode::EMPTY_DATASET,
                              "dataset4d.json: no cameras defined", manifest_path);
        }

        if (options.progress)
            options.progress(30.0f, "Parsing frames...");

        // ------------------------------------------------------------------ //
        // Parse frames into (time_index, cam_idx, image_path, mask_path).    //
        // ------------------------------------------------------------------ //
        // We first collect them, then reorganize into a 2D table.
        struct RawFrame {
            size_t time_index;
            size_t cam_idx;
            std::filesystem::path image_path;
            std::optional<std::filesystem::path> mask_path;
        };
        std::vector<RawFrame> raw_frames;
        raw_frames.reserve(j["frames"].size());

        size_t max_time_index = 0;

        for (size_t fi = 0; fi < j["frames"].size(); ++fi) {
            const auto& jf = j["frames"][fi];

            auto require_frame_field = [&](const char* field) -> Result<void> {
                if (!jf.contains(field)) {
                    return make_error(ErrorCode::INVALID_DATASET,
                                      std::format("frame[{}]: missing field '{}'", fi, field),
                                      manifest_path);
                }
                return {};
            };

            if (auto r = require_frame_field("time_index"); !r)
                return std::unexpected(r.error());
            if (auto r = require_frame_field("camera_id"); !r)
                return std::unexpected(r.error());
            if (auto r = require_frame_field("image_path"); !r)
                return std::unexpected(r.error());

            const size_t time_index = jf["time_index"].get<size_t>();
            const std::string cam_id = jf["camera_id"].get<std::string>();
            const std::string rel_image = jf["image_path"].get<std::string>();

            auto cam_it = cam_id_to_idx.find(cam_id);
            if (cam_it == cam_id_to_idx.end()) {
                return make_error(ErrorCode::INVALID_DATASET,
                                  std::format("frame[{}]: unknown camera_id '{}'", fi, cam_id),
                                  manifest_path);
            }

            const std::filesystem::path img_path = base_dir / lfs::core::utf8_to_path(rel_image);

            if (!options.validate_only && !safe_exists(img_path)) {
                return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                                  std::format("frame[{}]: image not found", fi), img_path);
            }

            std::optional<std::filesystem::path> mask_path;
            if (jf.contains("mask_path") && jf["mask_path"].is_string()) {
                const std::string rel_mask = jf["mask_path"].get<std::string>();
                if (!rel_mask.empty()) {
                    std::filesystem::path mp = base_dir / lfs::core::utf8_to_path(rel_mask);
                    if (!options.validate_only && !safe_exists(mp)) {
                        LOG_WARN("[4D] frame[{}]: optional mask not found at '{}', proceeding without mask",
                                 fi, lfs::core::path_to_utf8(mp));
                    } else {
                        mask_path = std::move(mp);
                    }
                }
            }

            max_time_index = std::max(max_time_index, time_index);
            raw_frames.push_back({time_index, cam_it->second, img_path, mask_path});
        }

        if (raw_frames.empty()) {
            return make_error(ErrorCode::EMPTY_DATASET,
                              "dataset4d.json: no frames defined", manifest_path);
        }

        if (options.progress)
            options.progress(60.0f, "Validating dataset completeness...");

        // ------------------------------------------------------------------ //
        // Build and validate the 2D frame table.                             //
        // ------------------------------------------------------------------ //
        const size_t num_time_steps = max_time_index + 1;
        const size_t num_cameras = cameras.size();

        // If no explicit timestamps were supplied, use frame indices as floats.
        if (timestamps.empty()) {
            timestamps.resize(num_time_steps);
            for (size_t i = 0; i < num_time_steps; ++i) {
                timestamps[i] = static_cast<float>(i);
            }
        } else if (timestamps.size() != num_time_steps) {
            return make_error(ErrorCode::INVALID_DATASET,
                              std::format("timestamps has {} entries but frames reference {} "
                                          "distinct time indices",
                                          timestamps.size(), num_time_steps),
                              manifest_path);
        }

        // Initialize the table with empty entries to detect missing frames.
        using FrameEntry = std::pair<std::filesystem::path, std::optional<std::filesystem::path>>;
        std::vector<std::vector<FrameEntry>> frame_table(
            num_time_steps, std::vector<FrameEntry>(num_cameras));

        // Track which cells have been populated.
        std::vector<std::vector<bool>> populated(
            num_time_steps, std::vector<bool>(num_cameras, false));

        for (const auto& rf : raw_frames) {
            if (rf.time_index >= num_time_steps) {
                return make_error(ErrorCode::INVALID_DATASET,
                                  std::format("time_index {} out of range [0,{})",
                                              rf.time_index, num_time_steps),
                                  manifest_path);
            }
            if (populated[rf.time_index][rf.cam_idx]) {
                return make_error(ErrorCode::INVALID_DATASET,
                                  std::format("Duplicate frame entry for (time={}, cam={})",
                                              rf.time_index, rf.cam_idx),
                                  manifest_path);
            }
            frame_table[rf.time_index][rf.cam_idx] = {rf.image_path, rf.mask_path};
            populated[rf.time_index][rf.cam_idx] = true;
        }

        // Validate completeness: every (time, camera) cell must be populated.
        std::vector<std::string> warnings;
        for (size_t t = 0; t < num_time_steps; ++t) {
            for (size_t c = 0; c < num_cameras; ++c) {
                if (!populated[t][c]) {
                    const std::string msg = std::format(
                        "Missing frame for camera '{}' at time step {}",
                        cameras[c]->image_name(), t);
                    LOG_ERROR("[4D] {}", msg);
                    return make_error(ErrorCode::INVALID_DATASET, msg, manifest_path);
                }
            }
        }

        if (options.validate_only) {
            if (options.progress)
                options.progress(100.0f, "4D dataset validation complete");
            LOG_INFO("[4D] Validation OK: {} cameras x {} time steps", num_cameras, num_time_steps);
            const auto end_time = std::chrono::high_resolution_clock::now();
            return LoadResult{
                .data = Loaded4DDataset{
                    .cameras = std::move(cameras),
                    .timestamps = std::move(timestamps),
                    .frames = {}},
                .loader_used = name(),
                .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                .warnings = {"Validation mode - frames not loaded"}};
        }

        if (options.progress)
            options.progress(90.0f, "Assembling dataset...");

        const auto end_time = std::chrono::high_resolution_clock::now();

        LOG_INFO("[4D] Loaded dataset: {} cameras x {} time steps from '{}'",
                 num_cameras, num_time_steps, lfs::core::path_to_utf8(base_dir));

        return LoadResult{
            .data = Loaded4DDataset{
                .cameras = std::move(cameras),
                .timestamps = std::move(timestamps),
                .frames = std::move(frame_table)},
            .loader_used = name(),
            .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
            .warnings = std::move(warnings)};
    }

} // namespace lfs::io
