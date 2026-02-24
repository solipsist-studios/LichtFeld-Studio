/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"
#include "io/error.hpp"
#include <chrono>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

// Forward declarations only - hide implementation details
namespace lfs::core {
    class Camera;
    class SplatData;
    struct PointCloud;
    struct MeshData;
} // namespace lfs::core

namespace lfs::io {

    // Import types from lfs::core for convenience
    using lfs::core::MeshData;
    using lfs::core::PointCloud;
    using lfs::core::SplatData;
    using lfs::core::Tensor;

    // Progress callback type
    using ProgressCallback = std::function<void(float percentage, const std::string& message)>;

    // Dataset type enum
    enum class DatasetType {
        Unknown,
        COLMAP,
        Transforms,
        Sequence  // 4D multi-camera sequence dataset (dataset4d.json)
    };

    // Public types that clients need
    struct LoadOptions {
        int resize_factor = -1;
        int max_width = 3840;
        std::string images_folder = "images";
        bool validate_only = false;
        ProgressCallback progress = nullptr;
    };

    struct LoadedScene {
        std::vector<std::shared_ptr<lfs::core::Camera>> cameras;
        std::shared_ptr<PointCloud> point_cloud;
    };

    /**
     * @brief A time-indexed multi-camera dataset for 4D training (OMG4/4D support).
     *
     * On-disk layout (identified by a "dataset4d.json" manifest):
     *
     *   <root>/
     *     dataset4d.json          # Manifest: cameras, timestamps, frame paths
     *     images/
     *       cam_000/              # One sub-directory per camera
     *         frame_0000.jpg      # Frames named frame_<NNNN>.<ext>
     *         frame_0001.jpg
     *       cam_001/
     *         ...
     *     masks/                  # Optional: same layout as images/
     *       cam_000/
     *         frame_0000.png
     *
     * dataset4d.json schema (version 1):
     * {
     *   "version": 1,
     *   "timestamps": [0.0, 0.033, ...],   // seconds; if absent, use frame indices
     *   "cameras": [
     *     {
     *       "id": "cam_000",
     *       "width": 1920, "height": 1080,
     *       "focal_x": 1000.0, "focal_y": 1000.0,
     *       "center_x": 960.0, "center_y": 540.0,
     *       "R": [[1,0,0],[0,1,0],[0,0,1]],   // 3x3 rotation matrix (row-major)
     *       "T": [0.0, 0.0, 0.0]              // translation vector
     *     }
     *   ],
     *   "frames": [
     *     {
     *       "time_index": 0,
     *       "camera_id": "cam_000",
     *       "image_path": "images/cam_000/frame_0000.jpg",
     *       "mask_path": "masks/cam_000/frame_0000.png"   // optional
     *     }
     *   ]
     * }
     *
     * Invariants enforced by the loader:
     *   - Every camera has exactly one frame per time step.
     *   - timestamps are monotonically increasing.
     *   - All image paths exist on disk (missing paths produce an error).
     */
    struct Loaded4DDataset {
        /// Fixed cameras (intrinsics/extrinsics constant across all time steps).
        std::vector<std::shared_ptr<lfs::core::Camera>> cameras;

        /// Discrete time steps in seconds (or frame indices when no timestamps supplied).
        std::vector<float> timestamps;

        /// Frame table: frames[time_idx][cam_idx] = {image_path, optional mask_path}.
        /// Dimensions: frames.size() == timestamps.size(),
        ///             frames[t].size() == cameras.size() for all t.
        std::vector<std::vector<std::pair<std::filesystem::path,
                                          std::optional<std::filesystem::path>>>> frames;
    };

    struct LoadResult {
        std::variant<std::shared_ptr<SplatData>, LoadedScene,
                     std::shared_ptr<MeshData>, Loaded4DDataset> data;
        Tensor scene_center;
        bool images_have_alpha = false;
        std::string loader_used;
        std::chrono::milliseconds load_time{0};
        std::vector<std::string> warnings;
    };

    /**
     * @brief Main loader interface - the ONLY public API for the loader module
     *
     * This class provides a clean facade over all loading functionality.
     * All implementation details are hidden behind this interface.
     */
    class LFS_IO_API Loader {
    public:
        /**
         * @brief Create a loader instance
         */
        static std::unique_ptr<Loader> create();

        /**
         * @brief Quick check if path contains a dataset (vs single file like PLY)
         * @param path Directory or file to check
         * @return true if dataset, false if single file or not loadable
         */
        static bool isDatasetPath(const std::filesystem::path& path);

        /**
         * @brief Check if path is a COLMAP sparse reconstruction folder
         * @param path Directory to check
         * @return true if directory contains cameras.bin/txt and images.bin/txt
         *
         * This can detect sparse COLMAP folders for camera-only imports
         * (where images folder may not exist).
         */
        static bool isColmapSparsePath(const std::filesystem::path& path);

        /**
         * @brief Determine the type of dataset at the given path
         * @param path Directory or file to check
         * @return DatasetType enum value
         */
        static DatasetType getDatasetType(const std::filesystem::path& path);

        /**
         * @brief Load data from any supported format
         * @param path File or directory to load
         * @param options Loading options
         * @return LoadResult on success, Error on failure (path not found, invalid dataset, etc.)
         */
        [[nodiscard]] virtual Result<LoadResult> load(
            const std::filesystem::path& path,
            const LoadOptions& options = {}) = 0;

        /**
         * @brief Check if a path can be loaded
         * @param path File or directory to check
         * @return true if the path can be loaded
         */
        virtual bool canLoad(const std::filesystem::path& path) const = 0;

        /**
         * @brief Get list of supported formats
         * @return Human-readable list of supported formats
         */
        virtual std::vector<std::string> getSupportedFormats() const = 0;

        /**
         * @brief Get list of supported file extensions
         * @return List of extensions (e.g., ".ply", ".json")
         */
        virtual std::vector<std::string> getSupportedExtensions() const = 0;

        virtual ~Loader() = default;
    };

    // PLY point cloud utilities

    /// Check if PLY contains Gaussian splat properties (opacity, scaling, rotation)
    /// Returns false for simple point clouds (xyz + colors only)
    LFS_IO_API bool is_gaussian_splat_ply(const std::filesystem::path& filepath);

    /// Load PLY as simple point cloud (xyz + optional colors)
    /// Use this for PLY files that are NOT Gaussian splats
    LFS_IO_API std::expected<PointCloud, std::string> load_ply_point_cloud(const std::filesystem::path& filepath);

} // namespace lfs::io