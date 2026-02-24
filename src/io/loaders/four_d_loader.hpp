/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/loader_interface.hpp"

namespace lfs::io {

    /**
     * @brief Loader for 4D multi-camera datasets (OMG4 Milestone 1).
     *
     * Detects directories that contain a "dataset4d.json" manifest and loads
     * them into a Loaded4DDataset.  See Loaded4DDataset in loader.hpp for the
     * full on-disk format specification.
     */
    class FourDLoader : public IDataLoader {
    public:
        FourDLoader() = default;
        ~FourDLoader() override = default;

        [[nodiscard]] Result<LoadResult> load(
            const std::filesystem::path& path,
            const LoadOptions& options = {}) override;

        bool canLoad(const std::filesystem::path& path) const override;
        std::string name() const override;
        std::vector<std::string> supportedExtensions() const override;
        int priority() const override;
    };

} // namespace lfs::io
