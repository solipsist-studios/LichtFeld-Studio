/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file omg4_loader.cpp
 * @brief OMG4 4D Gaussian Splat model loader.
 *
 * Implements loaders for:
 *   - .pth PyTorch checkpoint files containing 4D Gaussian parameters
 *   - .xz LZMA-compressed OMG4 format (stub: delegates to Python plugin)
 *
 * The .pth format is a ZIP archive containing pickle + raw tensor data.
 * We parse the ZIP index to detect the 4D extension fields and load them.
 *
 * Reference: https://arxiv.org/html/2510.03857v1
 */

#include "omg4_loader.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <string>

namespace lfs::io {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;

    namespace {

        // -----------------------------------------------------------------
        // Minimal ZIP central-directory parser to list .pth tensor files.
        // PyTorch .pth files are ZIP64 archives; we only need the file list
        // to know whether the 4D keys are present.
        // -----------------------------------------------------------------

        constexpr uint32_t ZIP_LOCAL_FILE_SIGNATURE = 0x04034b50;
        constexpr uint32_t ZIP_CENTRAL_DIR_SIGNATURE = 0x02014b50;
        constexpr uint32_t ZIP_END_OF_CENTRAL_DIR_SIGNATURE = 0x06054b50;

        bool starts_with(const std::string& s, const std::string& prefix) {
            return s.size() >= prefix.size() &&
                   s.substr(0, prefix.size()) == prefix;
        }

        /// Returns a list of filenames in the ZIP archive.
        std::vector<std::string> list_zip_entries(const std::filesystem::path& path) {
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f.is_open())
                return {};

            const auto file_size = f.tellg();
            if (file_size < 22)
                return {};

            // Scan backwards for the End-of-Central-Directory record
            const auto scan_size = std::min<std::streamoff>(65536 + 22, file_size);
            const auto scan_start = file_size - scan_size;
            f.seekg(scan_start);

            std::vector<uint8_t> buf(static_cast<size_t>(scan_size));
            f.read(reinterpret_cast<char*>(buf.data()), scan_size);

            std::streamoff eocd_offset = -1;
            for (std::streamoff i = static_cast<std::streamoff>(buf.size()) - 22; i >= 0; --i) {
                uint32_t sig;
                std::memcpy(&sig, buf.data() + i, 4);
                if (sig == ZIP_END_OF_CENTRAL_DIR_SIGNATURE) {
                    eocd_offset = scan_start + i;
                    break;
                }
            }
            if (eocd_offset < 0)
                return {};

            // Read EOCD
            f.seekg(eocd_offset + 16);
            uint32_t cd_offset = 0;
            f.read(reinterpret_cast<char*>(&cd_offset), 4);

            // Iterate central directory entries
            std::vector<std::string> entries;
            f.seekg(cd_offset);

            while (f.good()) {
                uint32_t sig = 0;
                f.read(reinterpret_cast<char*>(&sig), 4);
                if (sig != ZIP_CENTRAL_DIR_SIGNATURE)
                    break;

                f.seekg(24, std::ios::cur); // skip fields to filename length
                uint16_t fname_len = 0, extra_len = 0, comment_len = 0;
                f.read(reinterpret_cast<char*>(&fname_len), 2);
                f.read(reinterpret_cast<char*>(&extra_len), 2);
                f.read(reinterpret_cast<char*>(&comment_len), 2);
                f.seekg(12, std::ios::cur); // skip to filename

                std::string fname(fname_len, '\0');
                f.read(fname.data(), fname_len);
                f.seekg(extra_len + comment_len, std::ios::cur);

                entries.push_back(std::move(fname));
            }

            return entries;
        }

        /// Check whether a list of ZIP entries contains the 4D temporal keys.
        bool has_4d_keys(const std::vector<std::string>& entries) {
            bool has_t = false, has_st = false, has_rr = false;
            for (const auto& e : entries) {
                // PyTorch saves tensors with names like "archive/data/<key>"
                if (e.find("/t.") != std::string::npos ||
                    e.find("/t ") != std::string::npos ||
                    (e.find("/t") != std::string::npos && e.size() > e.find("/t") + 2 &&
                     e[e.find("/t") + 2] == '\0'))
                    has_t = true;
                if (e.find("scaling_t") != std::string::npos)
                    has_st = true;
                if (e.find("rotation_r") != std::string::npos)
                    has_rr = true;
            }
            return has_t && has_st && has_rr;
        }

        /// Create a placeholder zero tensor for a given shape when actual data is unavailable.
        Tensor zero_placeholder(std::initializer_list<long> shape, Device dev = Device::CUDA) {
            return Tensor::zeros(std::vector<long>(shape), dev, DataType::Float32);
        }

    } // anonymous namespace

    // -----------------------------------------------------------------

    bool is_omg4_file(const std::filesystem::path& path) {
        const auto ext = path.extension().string();
        if (ext == ".xz") {
            // Check LZMA magic bytes: FD 37 7A 58 5A 00
            std::ifstream f(path, std::ios::binary);
            if (!f.is_open())
                return false;
            uint8_t magic[6];
            f.read(reinterpret_cast<char*>(magic), 6);
            return f.gcount() == 6 && magic[0] == 0xFD && magic[1] == 0x37 &&
                   magic[2] == 0x7A && magic[3] == 0x58 && magic[4] == 0x5A &&
                   magic[5] == 0x00;
        }
        if (ext == ".pth") {
            const auto entries = list_zip_entries(path);
            return has_4d_keys(entries);
        }
        return false;
    }

    // -----------------------------------------------------------------

    std::expected<std::unique_ptr<lfs::core::SplatData4D>, std::string>
    load_omg4_checkpoint(const std::filesystem::path& path) {
        LOG_INFO("Loading OMG4 checkpoint: {}", lfs::core::path_to_utf8(path));

        if (!std::filesystem::exists(path)) {
            return std::unexpected(
                std::format("OMG4 checkpoint not found: {}", lfs::core::path_to_utf8(path)));
        }

        // Verify this looks like a 4D .pth file
        const auto entries = list_zip_entries(path);
        if (entries.empty()) {
            return std::unexpected(
                std::format("Not a valid ZIP/PyTorch file: {}", lfs::core::path_to_utf8(path)));
        }
        if (!has_4d_keys(entries)) {
            return std::unexpected(std::format(
                "PyTorch checkpoint does not contain 4D extension keys (t, scaling_t, rotation_r): {}",
                lfs::core::path_to_utf8(path)));
        }

        // ---------------------------------------------------------------
        // Full PyTorch tensor deserialization requires either:
        //   1. A C++ pickle/storage parser, or
        //   2. Delegating to the Python plugin (omg4_loader.py).
        //
        // For M1, we delegate to the Python plugin via a well-defined
        // protocol. The Python plugin loads the checkpoint and writes the
        // tensors as raw binary files to a temp directory, which we then
        // load here.
        //
        // As a fallback, return a descriptive error so the caller knows
        // to use the Python plugin path.
        // ---------------------------------------------------------------
        LOG_WARN("OMG4 checkpoint loader: native tensor deserialization not yet implemented. "
                 "Please use the Python plugin (src/python/lfs_plugins/omg4_loader.py) "
                 "to convert the checkpoint first.");

        return std::unexpected(std::format(
            "Native .pth loader not yet implemented for OMG4 checkpoint: {}. "
            "Use the Python plugin omg4_loader.py to load this file.",
            lfs::core::path_to_utf8(path)));
    }

    // -----------------------------------------------------------------

    std::expected<std::unique_ptr<lfs::core::SplatData4D>, std::string>
    load_omg4_compressed(const std::filesystem::path& path) {
        LOG_INFO("Loading OMG4 compressed model: {}", lfs::core::path_to_utf8(path));

        if (!std::filesystem::exists(path)) {
            return std::unexpected(
                std::format("OMG4 compressed file not found: {}", lfs::core::path_to_utf8(path)));
        }

        // Verify LZMA magic
        {
            std::ifstream f(path, std::ios::binary);
            uint8_t magic[6];
            f.read(reinterpret_cast<char*>(magic), 6);
            if (f.gcount() < 6 || magic[0] != 0xFD || magic[1] != 0x37 ||
                magic[2] != 0x7A || magic[3] != 0x58 || magic[4] != 0x5A || magic[5] != 0x00) {
                return std::unexpected(std::format(
                    "File does not appear to be LZMA-compressed (.xz): {}",
                    lfs::core::path_to_utf8(path)));
            }
        }

        // ---------------------------------------------------------------
        // The compressed format requires:
        //   1. LZMA decompression (xz-utils / liblzma)
        //   2. Python pickle deserialization
        //   3. Huffman decode of SVQ indices
        //   4. SVQ codebook lookup to reconstruct rotation/scale/appearance
        //   5. tiny-cuda-nn MLP forward pass for residual appearance
        //
        // For M1, this is delegated to the Python plugin.
        // ---------------------------------------------------------------
        LOG_WARN("OMG4 compressed loader: full SVQ/Huffman/MLP decoding not yet implemented. "
                 "Please use the Python plugin (src/python/lfs_plugins/omg4_loader.py) "
                 "to decode this file.");

        return std::unexpected(std::format(
            "Native .xz loader not yet implemented for OMG4 compressed model: {}. "
            "Use the Python plugin omg4_loader.py to load this file.",
            lfs::core::path_to_utf8(path)));
    }

} // namespace lfs::io
