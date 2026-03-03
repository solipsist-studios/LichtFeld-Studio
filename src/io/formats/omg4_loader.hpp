/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/splat_data_4d.hpp"
#include <expected>
#include <filesystem>
#include <memory>
#include <string>

namespace lfs::io {

    /**
     * @brief Load an OMG4 PyTorch checkpoint file (.pth) containing 4D Gaussian parameters.
     *
     * PyTorch .pth files are ZIP archives containing pickle + tensor data.
     * This loader handles .pth files that contain the 4D extension fields:
     *   - "xyz"          : positions [N, 3]
     *   - "features_dc"  : DC SH coefficients [N, 1, 3]
     *   - "features_rest": Higher-order SH [N, K, 3] (may be absent)
     *   - "scaling"      : 3D log-scale [N, 3]
     *   - "rotation"     : 3D rotation quaternion [N, 4]
     *   - "opacity"      : raw opacity logit [N, 1]
     *   - "t"            : temporal center [N, 1]          (4D extension)
     *   - "scaling_t"    : temporal log-scale [N, 1]       (4D extension)
     *   - "rotation_r"   : right-isoclinic quaternion [N, 4] (4D extension)
     *
     * @param path  Path to the .pth checkpoint file.
     * @return Loaded SplatData4D on success, error string on failure.
     */
    std::expected<std::unique_ptr<lfs::core::SplatData4D>, std::string>
    load_omg4_checkpoint(const std::filesystem::path& path);

    /**
     * @brief Load an OMG4 compressed model file (.xz).
     *
     * OMG4 compressed format is LZMA-compressed pickle containing SVQ-encoded
     * Gaussian attributes and tiny-cuda-nn MLP weights.
     *
     * @note Full SVQ/Huffman decoding and MLP evaluation are not yet implemented
     *       in this initial milestone. This function currently returns an error
     *       indicating that Python-side decoding via the omg4_loader plugin is
     *       required.
     *
     * @param path  Path to the .xz compressed model file.
     * @return Loaded SplatData4D on success, error string on failure.
     */
    std::expected<std::unique_ptr<lfs::core::SplatData4D>, std::string>
    load_omg4_compressed(const std::filesystem::path& path);

    /**
     * @brief Check whether a file appears to be an OMG4 model (either .pth or .xz).
     *
     * For .pth files, looks for the 4D temporal keys ("t", "scaling_t", "rotation_r").
     * For .xz files, checks the file magic.
     */
    bool is_omg4_file(const std::filesystem::path& path);

} // namespace lfs::io
