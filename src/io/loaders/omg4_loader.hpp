/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/loader_interface.hpp"

namespace lfs::io {

    /**
     * @brief Loader for OMG4 4D Gaussian Splat model files.
     *
     * Supports:
     *   - .pth  PyTorch checkpoint files containing 4D Gaussian parameters
     *   - .xz   LZMA-compressed OMG4 format
     *
     * Reference: https://arxiv.org/html/2510.03857v1
     */
    class Omg4Loader : public IDataLoader {
    public:
        Omg4Loader() = default;
        ~Omg4Loader() override = default;

        [[nodiscard]] Result<LoadResult> load(
            const std::filesystem::path& path,
            const LoadOptions& options = {}) override;

        bool canLoad(const std::filesystem::path& path) const override;
        std::string name() const override;
        std::vector<std::string> supportedExtensions() const override;
        int priority() const override;
    };

} // namespace lfs::io
