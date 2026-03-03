/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "omg4_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "formats/omg4_loader.hpp"
#include "io/error.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>

namespace lfs::io {

    Result<LoadResult> Omg4Loader::load(
        const std::filesystem::path& path,
        const LoadOptions& options) {

        LOG_TIMER("OMG4 Loading");
        auto start_time = std::chrono::high_resolution_clock::now();

        if (options.progress) {
            options.progress(0.0f, "Loading OMG4 model...");
        }

        if (!std::filesystem::exists(path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "OMG4 file does not exist", path);
        }

        if (options.validate_only) {
            if (!is_omg4_file(path)) {
                return make_error(ErrorCode::INVALID_HEADER,
                                  "File does not appear to be a valid OMG4 model", path);
            }
            LoadResult result;
            result.data = std::shared_ptr<SplatData>{};
            result.scene_center = lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU);
            result.loader_used = name();
            result.load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start_time);
            return result;
        }

        const auto ext = path.extension().string();
        std::string lower_ext = ext;
        std::transform(lower_ext.begin(), lower_ext.end(), lower_ext.begin(), ::tolower);

        std::expected<std::unique_ptr<lfs::core::SplatData4D>, std::string> model_result;
        if (lower_ext == ".xz") {
            if (options.progress)
                options.progress(10.0f, "Decompressing OMG4 model...");
            model_result = load_omg4_compressed(path);
        } else {
            // .pth
            if (options.progress)
                options.progress(10.0f, "Reading OMG4 checkpoint...");
            model_result = load_omg4_checkpoint(path);
        }

        if (!model_result) {
            return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                              std::format("OMG4 load failed: {}", model_result.error()), path);
        }

        if (options.progress)
            options.progress(100.0f, "OMG4 model loaded");

        auto end_time = std::chrono::high_resolution_clock::now();

        // Wrap the SplatData4D as a SplatData shared_ptr for the LoadResult.
        // The caller can dynamic_cast to SplatData4D* to access 4D fields.
        auto model_ptr = std::shared_ptr<lfs::core::SplatData4D>(std::move(*model_result));

        LoadResult result{
            .data = std::static_pointer_cast<SplatData>(model_ptr),
            .scene_center = lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            .loader_used = name(),
            .load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time),
            .warnings = {}};

        LOG_INFO("OMG4 model loaded successfully in {}ms",
                 result.load_time.count());
        return result;
    }

    bool Omg4Loader::canLoad(const std::filesystem::path& path) const {
        if (!std::filesystem::exists(path) || !std::filesystem::is_regular_file(path))
            return false;

        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".xz")
            return is_omg4_file(path);
        if (ext == ".pth")
            return is_omg4_file(path);
        return false;
    }

    std::string Omg4Loader::name() const {
        return "OMG4";
    }

    std::vector<std::string> Omg4Loader::supportedExtensions() const {
        return {".pth", ".PTH", ".xz", ".XZ"};
    }

    int Omg4Loader::priority() const {
        return 20; // High priority for OMG4-specific formats
    }

} // namespace lfs::io
