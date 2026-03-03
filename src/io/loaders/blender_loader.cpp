/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/loaders/blender_loader.hpp"
#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/point_cloud.hpp"
#include "formats/transforms.hpp"
#include "io/error.hpp"
#include "io/filesystem_utils.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <nlohmann/json.hpp>
#include <numbers>

namespace lfs::io {

    // Import types from lfs::core for convenience
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::PointCloud;
    using lfs::core::Tensor;

    namespace {
        constexpr std::array MASK_FOLDERS = {"masks", "mask", "segmentation"};
        constexpr std::array MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".mask.png"};

        // ---------------------------------------------------------------
        // Helpers shared between 3D and 4D loading paths
        // ---------------------------------------------------------------

        /// Tensor to glm::mat4 (row-major storage in Tensor, column-major in glm)
        glm::mat4 tensor_to_mat4(const Tensor& t) {
            glm::mat4 mat;
            const float* data = t.ptr<float>();
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    mat[j][i] = data[i * 4 + j];
            return mat;
        }

        Tensor mat4_to_tensor(const glm::mat4& mat) {
            Tensor t = Tensor::empty({4, 4}, Device::CPU, DataType::Float32);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    t[i][j] = mat[j][i];
            return t;
        }

        Tensor createYRotationMatrix(float angle_radians) {
            Tensor rot = Tensor::eye(4, Device::CPU);
            float c = std::cos(angle_radians);
            float s = std::sin(angle_radians);
            rot[0][0] = c;  rot[0][2] = s;
            rot[2][0] = -s; rot[2][2] = c;
            return rot;
        }

        /// Convert a 4×4 c2w matrix from Blender axes to R,T (COLMAP convention).
        /// Mirrors the logic in transforms.cpp::read_transforms_cameras_and_images.
        std::pair<Tensor, Tensor> c2w_json_to_RT(const nlohmann::json& transform_matrix) {
            Tensor c2w = Tensor::empty({4, 4}, Device::CPU, DataType::Float32);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    c2w[i][j] = float(transform_matrix[i][j]);

            // Blender → COLMAP axis flip: c2w[:3, 1:3] *= -1
            float* d = c2w.ptr<float>();
            for (int i = 0; i < 3; ++i) {
                d[i * 4 + 1] *= -1.0f;
                d[i * 4 + 2] *= -1.0f;
            }

            glm::mat4 w2c_glm = glm::inverse(tensor_to_mat4(c2w));
            Tensor w2c = mat4_to_tensor(w2c_glm);
            Tensor fixMat = createYRotationMatrix(static_cast<float>(std::numbers::pi));
            w2c = w2c.mm(fixMat);

            Tensor R = w2c.slice(0, 0, 3).slice(1, 0, 3).contiguous();
            Tensor T = w2c.slice(0, 0, 3).slice(1, 3, 4).squeeze(1).contiguous();
            return {R, T};
        }

        /// Read per-frame intrinsics, falling back to top-level values.
        struct FrameIntrinsics {
            float fl_x, fl_y, cx, cy;
            int w, h;
            float k1 = 0, k2 = 0, k3 = 0, p1 = 0, p2 = 0;
        };

        FrameIntrinsics read_intrinsics(const nlohmann::json& frame,
                                        const nlohmann::json& top) {
            auto get_f = [&](const char* key, float fallback) -> float {
                if (frame.contains(key) && frame[key].is_number()) return float(frame[key]);
                if (top.contains(key) && top[key].is_number()) return float(top[key]);
                return fallback;
            };
            auto get_i = [&](const char* key, int fallback) -> int {
                if (frame.contains(key) && frame[key].is_number()) return int(frame[key]);
                if (top.contains(key) && top[key].is_number()) return int(top[key]);
                return fallback;
            };

            FrameIntrinsics fi;
            fi.w  = get_i("w", -1);
            fi.h  = get_i("h", -1);
            fi.fl_x = get_f("fl_x", -1.0f);
            fi.fl_y = get_f("fl_y", -1.0f);
            fi.cx = get_f("cx", fi.w > 0 ? 0.5f * fi.w : -1.0f);
            fi.cy = get_f("cy", fi.h > 0 ? 0.5f * fi.h : -1.0f);
            fi.k1 = get_f("k1", 0.0f);
            fi.k2 = get_f("k2", 0.0f);
            fi.k3 = get_f("k3", 0.0f);
            fi.p1 = get_f("p1", 0.0f);
            fi.p2 = get_f("p2", 0.0f);
            return fi;
        }

        /// List image files in a directory, sorted lexicographically.
        std::vector<std::filesystem::path> list_images_sorted(const std::filesystem::path& dir) {
            std::vector<std::filesystem::path> files;
            std::error_code ec;
            for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
                if (!ec && entry.is_regular_file() && is_image_file(entry.path())) {
                    files.push_back(entry.path());
                }
            }
            std::sort(files.begin(), files.end());
            return files;
        }

        /// Check whether @p j qualifies as a 4D transforms file.
        /// Conditions: non-empty "frames", all frames have "camera_label",
        /// and the directory of the first frame's file_path has >1 image file.
        bool is_4d_transforms(const nlohmann::json& j,
                               const std::filesystem::path& base_dir) {
            if (!j.contains("frames") || !j["frames"].is_array()) return false;
            const auto& frames = j["frames"];
            if (frames.empty()) return false;

            // All frames must have camera_label
            for (const auto& f : frames) {
                if (!f.contains("camera_label") || !f["camera_label"].is_string())
                    return false;
            }

            // The first frame's image directory must have more than one image
            if (!frames[0].contains("file_path") || !frames[0]["file_path"].is_string())
                return false;

            const auto fp = lfs::core::utf8_to_path(frames[0]["file_path"].get<std::string>());
            const auto img_dir = (base_dir / fp).parent_path();
            if (!safe_exists(img_dir) || !safe_is_directory(img_dir)) return false;

            const auto imgs = list_images_sorted(img_dir);
            return imgs.size() > 1;
        }

        /// Load a 4D dataset from a transforms.json that passes is_4d_transforms().
        Result<LoadResult> load_4d_from_transforms(
            const std::filesystem::path& transforms_file,
            const nlohmann::json& j,
            const LoadOptions& options) {

            const auto start_time = std::chrono::high_resolution_clock::now();
            const std::filesystem::path base_dir = transforms_file.parent_path();

            if (options.progress) options.progress(0.0f, "Loading 4D sequence...");

            // -----------------------------------------------------------
            // 1. Parse frame entries → one Camera per camera_label
            // -----------------------------------------------------------
            struct ParsedFrame {
                std::string camera_label;
                std::filesystem::path img_dir; // directory that holds this camera's images
                Tensor R, T;
                FrameIntrinsics intr;
            };

            std::vector<std::string> ordered_labels; // first-seen order
            std::unordered_map<std::string, ParsedFrame> by_label;

            const auto& frames_json = j["frames"];
            for (size_t fi = 0; fi < frames_json.size(); ++fi) {
                const auto& jf = frames_json[fi];

                if (!jf.contains("transform_matrix")) {
                    return make_error(ErrorCode::INVALID_DATASET,
                                      std::format("frame[{}]: missing 'transform_matrix'", fi),
                                      transforms_file);
                }

                const std::string label = jf["camera_label"].get<std::string>();

                if (by_label.count(label) == 0) {
                    // First occurrence – parse camera data
                    ParsedFrame pf;
                    pf.camera_label = label;

                    // Resolve image directory
                    if (!jf.contains("file_path") || !jf["file_path"].is_string()) {
                        return make_error(ErrorCode::INVALID_DATASET,
                                          std::format("frame[{}]: missing 'file_path'", fi),
                                          transforms_file);
                    }
                    const auto fp = lfs::core::utf8_to_path(jf["file_path"].get<std::string>());
                    pf.img_dir = (base_dir / fp).parent_path();

                    // Parse transform
                    try {
                        auto [R, T] = c2w_json_to_RT(jf["transform_matrix"]);
                        pf.R = std::move(R);
                        pf.T = std::move(T);
                    } catch (const std::exception& e) {
                        return make_error(ErrorCode::INVALID_DATASET,
                                          std::format("frame[{}] '{}': bad transform_matrix: {}",
                                                      fi, label, e.what()),
                                          transforms_file);
                    }

                    // Intrinsics
                    pf.intr = read_intrinsics(jf, j);

                    ordered_labels.push_back(label);
                    by_label[label] = std::move(pf);
                }
                // Additional occurrences of the same camera_label are ignored
                // (only first extrinsics are used).
            }

            if (ordered_labels.empty()) {
                return make_error(ErrorCode::EMPTY_DATASET, "No cameras found", transforms_file);
            }

            if (options.progress) options.progress(20.0f, "Discovering time steps...");

            // -----------------------------------------------------------
            // 2. For each camera, discover sorted image files (= time steps)
            // -----------------------------------------------------------
            size_t num_time_steps = 0;
            std::vector<std::vector<std::filesystem::path>> cam_images; // [cam_idx][time_idx]
            cam_images.reserve(ordered_labels.size());

            for (size_t ci = 0; ci < ordered_labels.size(); ++ci) {
                const auto& pf = by_label[ordered_labels[ci]];
                if (!safe_exists(pf.img_dir) || !safe_is_directory(pf.img_dir)) {
                    return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                                      std::format("Camera '{}': image directory not found",
                                                  pf.camera_label),
                                      pf.img_dir);
                }

                auto imgs = list_images_sorted(pf.img_dir);
                if (imgs.empty()) {
                    return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                                      std::format("Camera '{}': no image files found",
                                                  pf.camera_label),
                                      pf.img_dir);
                }

                if (ci == 0) {
                    num_time_steps = imgs.size();
                } else if (imgs.size() != num_time_steps) {
                    return make_error(ErrorCode::INVALID_DATASET,
                                      std::format("Camera '{}' has {} time steps but camera '{}' "
                                                  "has {}; all cameras must match",
                                                  pf.camera_label, imgs.size(),
                                                  ordered_labels[0], num_time_steps),
                                      transforms_file);
                }

                cam_images.push_back(std::move(imgs));
            }

            if (options.progress) options.progress(50.0f, "Building camera objects...");

            // -----------------------------------------------------------
            // 3. Build Camera objects
            // -----------------------------------------------------------
            const size_t num_cams = ordered_labels.size();
            std::vector<std::shared_ptr<lfs::core::Camera>> cameras;
            cameras.reserve(num_cams);

            for (size_t ci = 0; ci < num_cams; ++ci) {
                const auto& pf = by_label[ordered_labels[ci]];
                const auto& intr = pf.intr;

                // Build distortion tensors
                bool is_distorted = (intr.k1 != 0.0f) || (intr.k2 != 0.0f) ||
                                    (intr.k3 != 0.0f) || (intr.p1 != 0.0f) || (intr.p2 != 0.0f);
                Tensor radial = is_distorted
                    ? Tensor::from_vector({intr.k1, intr.k2, intr.k3}, {3}, Device::CPU)
                    : Tensor::empty({0}, Device::CPU);
                Tensor tangential = is_distorted
                    ? Tensor::from_vector({intr.p1, intr.p2}, {2}, Device::CPU)
                    : Tensor::empty({0}, Device::CPU);

                auto cam = std::make_shared<lfs::core::Camera>(
                    pf.R,
                    pf.T,
                    intr.fl_x,
                    intr.fl_y,
                    intr.cx > 0 ? intr.cx : (intr.w > 0 ? 0.5f * static_cast<float>(intr.w) : 0.0f),
                    intr.cy > 0 ? intr.cy : (intr.h > 0 ? 0.5f * static_cast<float>(intr.h) : 0.0f),
                    radial,
                    tangential,
                    lfs::core::CameraModelType::PINHOLE,
                    pf.camera_label,
                    std::filesystem::path{},
                    std::filesystem::path{},
                    intr.w,
                    intr.h,
                    static_cast<int>(ci));

                cameras.push_back(std::move(cam));
            }

            if (options.progress) options.progress(70.0f, "Assembling frame table...");

            // -----------------------------------------------------------
            // 4. Build frame table: frames[time_idx][cam_idx]
            // -----------------------------------------------------------
            using FrameEntry = std::pair<std::filesystem::path,
                                         std::optional<std::filesystem::path>>;
            std::vector<std::vector<FrameEntry>> frame_table(
                num_time_steps, std::vector<FrameEntry>(num_cams));

            for (size_t ti = 0; ti < num_time_steps; ++ti) {
                for (size_t ci = 0; ci < num_cams; ++ci) {
                    frame_table[ti][ci] = {cam_images[ci][ti], std::nullopt};
                }
            }

            // -----------------------------------------------------------
            // 5. Timestamps = 0, 1, 2, …
            // -----------------------------------------------------------
            std::vector<float> timestamps(num_time_steps);
            for (size_t i = 0; i < num_time_steps; ++i)
                timestamps[i] = static_cast<float>(i);

            // -----------------------------------------------------------
            // 6. Optional fps warning
            // -----------------------------------------------------------
            std::vector<std::string> warnings;
            if (j.contains("fps") && j["fps"].is_number()) {
                warnings.push_back(
                    std::format("fps={} found in transforms.json (timestamps are frame indices; "
                                "apply fps for wall-clock time)",
                                float(j["fps"])));
            }

            const auto end_time = std::chrono::high_resolution_clock::now();

            LOG_INFO("[BlenderLoader-4D] Loaded {} cameras × {} time steps from '{}'",
                     num_cams, num_time_steps, lfs::core::path_to_utf8(transforms_file));

            return LoadResult{
                .data = Loaded4DDataset{
                    .cameras = std::move(cameras),
                    .timestamps = std::move(timestamps),
                    .frames = std::move(frame_table)},
                .loader_used = "BlenderLoader-4D",
                .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                .warnings = std::move(warnings)};
        }

    } // namespace

    static std::filesystem::path find_mask_path(const std::filesystem::path& base_path,
                                                const std::string& image_name) {
        const std::filesystem::path img_path = lfs::core::utf8_to_path(image_name);
        const std::filesystem::path stem_path = img_path.parent_path() / img_path.stem();

        for (const auto& folder : MASK_FOLDERS) {
            const std::filesystem::path mask_dir = base_path / folder;
            if (!safe_exists(mask_dir))
                continue;

            if (const auto exact = mask_dir / img_path; safe_exists(exact))
                return exact;

            if (auto found = find_path_ci(mask_dir, img_path); !found.empty())
                return found;

            for (const auto& ext : MASK_EXTENSIONS) {
                std::filesystem::path target_path = stem_path;
                target_path += ext;
                if (auto found = find_path_ci(mask_dir, target_path); !found.empty())
                    return found;
            }

            for (const auto& ext : MASK_EXTENSIONS) {
                std::filesystem::path target_path = img_path;
                target_path += ext;
                if (auto found = find_path_ci(mask_dir, target_path); !found.empty())
                    return found;
            }
        }
        return {};
    }

    Result<LoadResult> BlenderLoader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("Blender/NeRF Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        // Validate path exists
        if (!std::filesystem::exists(path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "Blender/NeRF dataset path does not exist", path);
        }

        // Report initial progress
        if (options.progress) {
            options.progress(0.0f, "Loading Blender/NeRF dataset...");
        }

        // Determine transforms file path
        std::filesystem::path transforms_file;

        if (std::filesystem::is_directory(path)) {
            // Look for transforms files in directory
            if (std::filesystem::exists(path / "transforms_train.json")) {
                transforms_file = path / "transforms_train.json";
                LOG_DEBUG("Found transforms_train.json");
            } else if (std::filesystem::exists(path / "transforms.json")) {
                transforms_file = path / "transforms.json";
                LOG_DEBUG("Found transforms.json");
            } else {
                return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                                  "No transforms file found (expected 'transforms.json' or 'transforms_train.json')", path);
            }
        } else if (path.extension() == ".json") {
            // Direct path to transforms file
            transforms_file = path;
            LOG_DEBUG("Using direct transforms file: {}", lfs::core::path_to_utf8(transforms_file));
        } else {
            return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                              "Path must be a directory or a JSON file", path);
        }

        // Validation only mode
        if (options.validate_only) {
            LOG_DEBUG("Validation only mode for Blender/NeRF: {}", lfs::core::path_to_utf8(transforms_file));
            // Check if the transforms file is valid JSON
            std::ifstream file;
            if (!lfs::core::open_file_for_read(transforms_file, file)) {
                return make_error(ErrorCode::PERMISSION_DENIED,
                                  "Cannot open transforms file for reading", transforms_file);
            }

            // Try to parse as JSON (basic validation)
            try {
                nlohmann::json j;
                file >> j;

                if (!j.contains("frames") || !j["frames"].is_array()) {
                    return make_error(ErrorCode::INVALID_DATASET,
                                      "Invalid transforms file: missing 'frames' array", transforms_file);
                }
            } catch (const std::exception& e) {
                return make_error(ErrorCode::MALFORMED_JSON,
                                  std::format("Invalid JSON: {}", e.what()), transforms_file);
            }

            if (options.progress) {
                options.progress(100.0f, "Blender/NeRF validation complete");
            }

            LOG_DEBUG("Blender/NeRF validation successful");

            auto end_time = std::chrono::high_resolution_clock::now();
            return LoadResult{
                .data = LoadedScene{
                    .cameras = {},
                    .point_cloud = nullptr},
                .scene_center = Tensor::zeros({3}, Device::CPU, DataType::Float32),
                .loader_used = name(),
                .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                .warnings = {"Validation mode - point cloud not loaded"}};
        }

        // Load the dataset
        if (options.progress) {
            options.progress(20.0f, "Reading transforms file...");
        }

        try {
            LOG_INFO("Loading Blender/NeRF dataset from: {}", lfs::core::path_to_utf8(transforms_file));

            // -----------------------------------------------------------
            // 4D detection: parse JSON and check for 4D signal before
            // delegating to the standard 3D path.
            // -----------------------------------------------------------
            {
                std::ifstream jf_stream;
                if (lfs::core::open_file_for_read(transforms_file, jf_stream)) {
                    try {
                        nlohmann::json j = nlohmann::json::parse(jf_stream, nullptr, true, true);
                        if (is_4d_transforms(j, transforms_file.parent_path())) {
                            return load_4d_from_transforms(transforms_file, j, options);
                        }
                    } catch (...) {
                        // JSON parse failure – fall through to 3D path which will give a proper error
                    }
                }
            }

            // Read transforms and create cameras
            auto [camera_infos, scene_center, train_val_split] = read_transforms_cameras_and_images(transforms_file);

            if (options.progress) {
                options.progress(40.0f, std::format("Creating {} cameras...", camera_infos.size()));
            }

            LOG_DEBUG("Creating {} camera objects", camera_infos.size());

            // Convert CameraData to Camera objects
            std::vector<std::shared_ptr<lfs::core::Camera>> cameras;
            cameras.reserve(camera_infos.size());

            // Get base path for mask lookup
            std::filesystem::path base_path = transforms_file.parent_path();

            for (size_t i = 0; i < camera_infos.size(); ++i) {
                const auto& info = camera_infos[i];

                try {
                    // Find mask path if available
                    std::filesystem::path mask_path = find_mask_path(base_path, info._image_name);

                    // Validate mask dimensions match image dimensions
                    if (!mask_path.empty()) {
                        auto [img_w, img_h, img_c] = lfs::core::get_image_info(info._image_path);
                        auto [mask_w, mask_h, mask_c] = lfs::core::get_image_info(mask_path);
                        if (img_w != mask_w || img_h != mask_h) {
                            return make_error(ErrorCode::MASK_SIZE_MISMATCH,
                                              std::format("Mask '{}' is {}x{} but image '{}' is {}x{}",
                                                          lfs::core::path_to_utf8(mask_path.filename()), mask_w, mask_h,
                                                          info._image_name, img_w, img_h),
                                              mask_path);
                        }
                    }

                    auto cam = std::make_shared<lfs::core::Camera>(
                        info._R,
                        info._T,
                        info._focal_x,
                        info._focal_y,
                        info._center_x,
                        info._center_y,
                        info._radial_distortion,
                        info._tangential_distortion,
                        info._camera_model_type,
                        info._image_name,
                        info._image_path,
                        mask_path,
                        info._width,
                        info._height,
                        static_cast<int>(i));

                    cameras.push_back(cam);
                } catch (const std::exception& e) {
                    LOG_ERROR("Failed to create camera {}: {}", i, e.what());
                    throw;
                }
            }

            bool images_have_alpha = false;
            if (!cameras.empty()) {
                try {
                    auto [w, h, c] = lfs::core::get_image_info(cameras[0]->image_path());
                    images_have_alpha = (c == 4);
                } catch (const std::exception&) {
                }
            }

            if (options.progress) {
                options.progress(60.0f, "Loading point cloud...");
            }

            // Check ply_file_path in transforms.json (nerfstudio format), fallback to pointcloud.ply
            std::filesystem::path pointcloud_path;
            if (std::ifstream file; lfs::core::open_file_for_read(transforms_file, file)) {
                try {
                    if (const auto json = nlohmann::json::parse(file, nullptr, true, true);
                        json.contains("ply_file_path")) {
                        pointcloud_path = base_path / lfs::core::utf8_to_path(json["ply_file_path"].get<std::string>());
                    }
                } catch (...) {
                    // Ignore parse errors - will fallback to default pointcloud.ply
                }
            }
            if (pointcloud_path.empty() || !std::filesystem::exists(pointcloud_path)) {
                pointcloud_path = base_path / "pointcloud.ply";
            }

            std::shared_ptr<PointCloud> point_cloud;
            std::vector<std::string> warnings;
            if (std::filesystem::exists(pointcloud_path)) {
                point_cloud = std::make_shared<PointCloud>(load_simple_ply_point_cloud(pointcloud_path));
                LOG_INFO("Loaded {} points from {}", point_cloud->size(),
                         lfs::core::path_to_utf8(pointcloud_path.filename()));
            } else {
                point_cloud = std::make_shared<PointCloud>(generate_random_point_cloud());
                LOG_WARN("No PLY found, using {} random points", point_cloud->size());
                warnings.emplace_back("No point cloud file found, using random initialization");
            }

            if (options.progress) {
                options.progress(100.0f, "Blender/NeRF loading complete");
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time);

            auto scene_center_cpu = scene_center.cpu();
            const float* sc_ptr = scene_center_cpu.template ptr<float>();

            size_t num_cameras = cameras.size();

            LoadResult result{
                .data = LoadedScene{
                    .cameras = std::move(cameras),
                    .point_cloud = std::move(point_cloud)},
                .scene_center = scene_center,
                .images_have_alpha = images_have_alpha,
                .loader_used = name(),
                .load_time = load_time,
                .warnings = std::move(warnings)};

            LOG_INFO("Blender/NeRF dataset loaded successfully in {}ms", load_time.count());
            LOG_INFO("  - {} cameras", num_cameras);
            LOG_DEBUG("  - Scene center: [{:.3f}, {:.3f}, {:.3f}]",
                      sc_ptr[0], sc_ptr[1], sc_ptr[2]);

            return result;

        } catch (const std::exception& e) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Failed to load Blender/NeRF dataset: {}", e.what()), path);
        }
    }

    bool BlenderLoader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path)) {
            return false;
        }

        if (std::filesystem::is_directory(path)) {
            // Check for transforms files in directory
            return std::filesystem::exists(path / "transforms.json") ||
                   std::filesystem::exists(path / "transforms_train.json");
        } else {
            // Check if it's a JSON file
            return path.extension() == ".json";
        }
    }

    std::string BlenderLoader::name() const {
        return "Blender/NeRF";
    }

    std::vector<std::string> BlenderLoader::supportedExtensions() const {
        return {".json"}; // Can load JSON files directly
    }

    int BlenderLoader::priority() const {
        return 5; // Medium priority
    }

} // namespace lfs::io