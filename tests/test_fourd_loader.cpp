/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_fourd_loader.cpp
 * @brief Unit and integration tests for the 4D dataset loader (FourDLoader).
 *
 * Tests cover:
 *   - Loader detection (canLoad)
 *   - Happy-path loading of a well-formed dataset4d.json
 *   - Validation of completeness (missing frames detected)
 *   - Timestamp monotonicity enforcement
 *   - SequenceDataset access patterns (time slice, (cam,time) pair, nearest time)
 */

#include "io/loader.hpp"
#include "io/loaders/fourd_loader.hpp"
#include "training/sequence_dataset.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace lfs::io;
using namespace lfs::training;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

    /// Write a string to a file (creating parent directories as needed).
    void write_file(const fs::path& path, const std::string& content) {
        fs::create_directories(path.parent_path());
        std::ofstream f(path, std::ios::binary);
        ASSERT_TRUE(f.is_open()) << "Cannot open " << path;
        f << content;
    }

    /// Build a minimal valid dataset4d.json string for @p num_cams cameras and
    /// @p num_times time steps.  Image files are given placeholder paths
    /// (validate_only mode does not check for on-disk existence).
    std::string make_manifest(int num_cams, int num_times,
                              bool include_timestamps = true,
                              bool include_masks = false) {
        std::string j = "{\n  \"version\": 1,\n";

        if (include_timestamps) {
            j += "  \"timestamps\": [";
            for (int t = 0; t < num_times; ++t) {
                j += std::format("{}{:.3f}", t > 0 ? "," : "", t * 0.033f);
            }
            j += "],\n";
        }

        j += "  \"cameras\": [\n";
        for (int c = 0; c < num_cams; ++c) {
            j += std::format(
                "    {{\n"
                "      \"id\": \"cam_{:03d}\",\n"
                "      \"width\": 640, \"height\": 480,\n"
                "      \"focal_x\": 320.0, \"focal_y\": 320.0,\n"
                "      \"center_x\": 320.0, \"center_y\": 240.0,\n"
                "      \"R\": [[1,0,0],[0,1,0],[0,0,1]],\n"
                "      \"T\": [0.0, 0.0, 0.0]\n"
                "    }}{}",
                c, c + 1 < num_cams ? "," : "");
            j += "\n";
        }
        j += "  ],\n";

        j += "  \"frames\": [\n";
        bool first = true;
        for (int t = 0; t < num_times; ++t) {
            for (int c = 0; c < num_cams; ++c) {
                if (!first)
                    j += ",\n";
                first = false;

                j += std::format(
                    "    {{\n"
                    "      \"time_index\": {},\n"
                    "      \"camera_id\": \"cam_{:03d}\",\n"
                    "      \"image_path\": \"images/cam_{:03d}/frame_{:04d}.jpg\"",
                    t, c, c, t);

                if (include_masks) {
                    j += std::format(
                        ",\n      \"mask_path\": \"masks/cam_{:03d}/frame_{:04d}.png\"", c, t);
                }

                j += "\n    }";
            }
        }
        j += "\n  ]\n}\n";
        return j;
    }

    /// Create a temporary directory, write dataset4d.json, and optionally touch
    /// the referenced image (and mask) files so existence checks pass.
    struct TmpDataset {
        fs::path root;

        explicit TmpDataset(const std::string& tag) {
            root = fs::temp_directory_path() / ("lfs_test_4d_" + tag);
            fs::remove_all(root);
            fs::create_directories(root);
        }

        ~TmpDataset() {
            std::error_code ec;
            fs::remove_all(root, ec);
        }

        void write_manifest(const std::string& content) {
            write_file(root / "dataset4d.json", content);
        }

        /// Touch the image files that would be referenced by make_manifest().
        void touch_images(int num_cams, int num_times) {
            for (int t = 0; t < num_times; ++t) {
                for (int c = 0; c < num_cams; ++c) {
                    const fs::path img = root / std::format("images/cam_{:03d}/frame_{:04d}.jpg", c, t);
                    fs::create_directories(img.parent_path());
                    std::ofstream f(img); // empty file is fine
                }
            }
        }
    };

} // namespace

// ---------------------------------------------------------------------------
// FourDLoader::canLoad
// ---------------------------------------------------------------------------

class FourDLoaderCanLoadTest : public ::testing::Test {
protected:
    FourDLoader loader;

    TmpDataset ds{"canload"};
};

TEST_F(FourDLoaderCanLoadTest, ReturnsFalseForNonExistentPath) {
    EXPECT_FALSE(loader.canLoad(ds.root / "does_not_exist"));
}

TEST_F(FourDLoaderCanLoadTest, ReturnsFalseForDirectoryWithoutManifest) {
    EXPECT_FALSE(loader.canLoad(ds.root));
}

TEST_F(FourDLoaderCanLoadTest, ReturnsTrueForDirectoryWithManifest) {
    ds.write_manifest("{}"); // content doesn't matter for canLoad
    EXPECT_TRUE(loader.canLoad(ds.root));
}

TEST_F(FourDLoaderCanLoadTest, ReturnsTrueForManifestFileDirect) {
    ds.write_manifest("{}");
    EXPECT_TRUE(loader.canLoad(ds.root / "dataset4d.json"));
}

// ---------------------------------------------------------------------------
// Validate-only mode
// ---------------------------------------------------------------------------

class FourDLoaderValidateTest : public ::testing::Test {
protected:
    FourDLoader loader;
    LoadOptions opts;

    void SetUp() override {
        opts.validate_only = true;
    }
};

TEST_F(FourDLoaderValidateTest, ValidManifestPassesValidation) {
    TmpDataset ds("validate_ok");
    ds.write_manifest(make_manifest(2, 3));

    auto result = loader.load(ds.root, opts);
    ASSERT_TRUE(result.has_value()) << result.error().format();
    EXPECT_EQ(result->loader_used, "4D");

    const auto& d4d = std::get<Loaded4DDataset>(result->data);
    EXPECT_EQ(d4d.cameras.size(), 2u);
    EXPECT_EQ(d4d.timestamps.size(), 3u);
    // In validate mode the frames table is empty
    EXPECT_TRUE(d4d.frames.empty());
}

TEST_F(FourDLoaderValidateTest, MissingFrameDetected) {
    TmpDataset ds("validate_miss");

    // Build a manifest where one frame is missing (cam_001 at t=1).
    std::string j = R"({
  "version": 1,
  "cameras": [
    {"id":"cam_000","width":640,"height":480,"focal_x":320,"focal_y":320,"center_x":320,"center_y":240,
     "R":[[1,0,0],[0,1,0],[0,0,1]],"T":[0,0,0]},
    {"id":"cam_001","width":640,"height":480,"focal_x":320,"focal_y":320,"center_x":320,"center_y":240,
     "R":[[1,0,0],[0,1,0],[0,0,1]],"T":[0,0,0]}
  ],
  "frames": [
    {"time_index":0,"camera_id":"cam_000","image_path":"images/c0_t0.jpg"},
    {"time_index":0,"camera_id":"cam_001","image_path":"images/c1_t0.jpg"},
    {"time_index":1,"camera_id":"cam_000","image_path":"images/c0_t1.jpg"}
  ]
})";
    ds.write_manifest(j);

    auto result = loader.load(ds.root, opts);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::INVALID_DATASET);
}

TEST_F(FourDLoaderValidateTest, NonMonotonicTimestampsRejected) {
    TmpDataset ds("validate_ts");

    std::string j = R"({
  "version": 1,
  "timestamps": [0.0, 0.5, 0.3],
  "cameras": [
    {"id":"cam_000","width":640,"height":480,"focal_x":320,"focal_y":320,"center_x":320,"center_y":240,
     "R":[[1,0,0],[0,1,0],[0,0,1]],"T":[0,0,0]}
  ],
  "frames": [
    {"time_index":0,"camera_id":"cam_000","image_path":"img0.jpg"},
    {"time_index":1,"camera_id":"cam_000","image_path":"img1.jpg"},
    {"time_index":2,"camera_id":"cam_000","image_path":"img2.jpg"}
  ]
})";
    ds.write_manifest(j);

    auto result = loader.load(ds.root, opts);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::INVALID_DATASET);
}

TEST_F(FourDLoaderValidateTest, DuplicateCameraIdRejected) {
    TmpDataset ds("validate_dup");

    std::string j = R"({
  "version": 1,
  "cameras": [
    {"id":"cam_000","width":1,"height":1,"focal_x":1,"focal_y":1,"center_x":0,"center_y":0,
     "R":[[1,0,0],[0,1,0],[0,0,1]],"T":[0,0,0]},
    {"id":"cam_000","width":1,"height":1,"focal_x":1,"focal_y":1,"center_x":0,"center_y":0,
     "R":[[1,0,0],[0,1,0],[0,0,1]],"T":[0,0,0]}
  ],
  "frames": [
    {"time_index":0,"camera_id":"cam_000","image_path":"img.jpg"}
  ]
})";
    ds.write_manifest(j);

    auto result = loader.load(ds.root, opts);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::INVALID_DATASET);
}

TEST_F(FourDLoaderValidateTest, MissingCameraFieldRejected) {
    TmpDataset ds("validate_field");

    // Missing "width"
    std::string j = R"({
  "version": 1,
  "cameras": [
    {"id":"cam_000","height":480,"focal_x":320,"focal_y":320,"center_x":320,"center_y":240,
     "R":[[1,0,0],[0,1,0],[0,0,1]],"T":[0,0,0]}
  ],
  "frames": [
    {"time_index":0,"camera_id":"cam_000","image_path":"img.jpg"}
  ]
})";
    ds.write_manifest(j);

    auto result = loader.load(ds.root, opts);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::INVALID_DATASET);
}

TEST_F(FourDLoaderValidateTest, ImplicitTimestampsFromFrameIndices) {
    TmpDataset ds("validate_implicit_ts");
    // No "timestamps" key → loader generates 0,1,2,...
    ds.write_manifest(make_manifest(1, 4, /*include_timestamps=*/false));

    auto result = loader.load(ds.root, opts);
    ASSERT_TRUE(result.has_value()) << result.error().format();

    const auto& d4d = std::get<Loaded4DDataset>(result->data);
    ASSERT_EQ(d4d.timestamps.size(), 4u);
    EXPECT_FLOAT_EQ(d4d.timestamps[0], 0.0f);
    EXPECT_FLOAT_EQ(d4d.timestamps[3], 3.0f);
}

// ---------------------------------------------------------------------------
// Full (non-validate) loading with actual image files on disk
// ---------------------------------------------------------------------------

class FourDLoaderLoadTest : public ::testing::Test {
protected:
    FourDLoader loader;
    LoadOptions opts; // default: validate_only = false
};

TEST_F(FourDLoaderLoadTest, LoadsDatasetCorrectly) {
    TmpDataset ds("load_ok");
    ds.write_manifest(make_manifest(3, 5));
    ds.touch_images(3, 5);

    auto result = loader.load(ds.root, opts);
    ASSERT_TRUE(result.has_value()) << result.error().format();

    const auto& d4d = std::get<Loaded4DDataset>(result->data);
    EXPECT_EQ(d4d.cameras.size(), 3u);
    EXPECT_EQ(d4d.timestamps.size(), 5u);
    EXPECT_EQ(d4d.frames.size(), 5u);
    for (const auto& row : d4d.frames) {
        EXPECT_EQ(row.size(), 3u);
        for (const auto& entry : row) {
            EXPECT_FALSE(entry.first.empty());
        }
    }
}

TEST_F(FourDLoaderLoadTest, MissingImageFileProducesError) {
    TmpDataset ds("load_missing");
    ds.write_manifest(make_manifest(1, 2));
    // Intentionally do NOT touch images → files don't exist

    auto result = loader.load(ds.root, opts);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::MISSING_REQUIRED_FILES);
}

TEST_F(FourDLoaderLoadTest, LoaderNameIs4D) {
    EXPECT_EQ(loader.name(), "4D");
}

// ---------------------------------------------------------------------------
// SequenceDataset access patterns
// ---------------------------------------------------------------------------

/// Build a small SequenceDataset in-memory for access pattern tests.
static std::shared_ptr<SequenceDataset> make_sequence_dataset(size_t num_cams, size_t num_times) {
    std::vector<std::shared_ptr<lfs::core::Camera>> cameras;
    cameras.reserve(num_cams);
    for (size_t c = 0; c < num_cams; ++c) {
        // Use default-constructed camera (enough for access tests)
        cameras.push_back(std::make_shared<lfs::core::Camera>());
    }

    std::vector<float> timestamps(num_times);
    for (size_t t = 0; t < num_times; ++t) {
        timestamps[t] = static_cast<float>(t) * 0.5f; // 0.0, 0.5, 1.0, ...
    }

    std::vector<std::vector<SequenceFrameEntry>> frames(num_times);
    for (size_t t = 0; t < num_times; ++t) {
        frames[t].resize(num_cams);
        for (size_t c = 0; c < num_cams; ++c) {
            frames[t][c].image_path = std::format("img_t{}_c{}.jpg", t, c);
        }
    }

    return std::make_shared<SequenceDataset>(
        std::move(cameras), std::move(timestamps), std::move(frames));
}

TEST(SequenceDatasetTest, NumCamerasAndTimesteps) {
    auto ds = make_sequence_dataset(4, 6);
    EXPECT_EQ(ds->num_cameras(), 4u);
    EXPECT_EQ(ds->num_timesteps(), 6u);
}

TEST(SequenceDatasetTest, GetTimestamp) {
    auto ds = make_sequence_dataset(2, 3);
    EXPECT_FLOAT_EQ(ds->get_timestamp(0), 0.0f);
    EXPECT_FLOAT_EQ(ds->get_timestamp(1), 0.5f);
    EXPECT_FLOAT_EQ(ds->get_timestamp(2), 1.0f);
}

TEST(SequenceDatasetTest, GetTimestampOutOfRangeThrows) {
    auto ds = make_sequence_dataset(1, 2);
    EXPECT_THROW(ds->get_timestamp(2), std::out_of_range);
}

TEST(SequenceDatasetTest, GetTimeSlice) {
    auto ds = make_sequence_dataset(3, 4);
    const auto& slice = ds->get_time_slice(2);
    ASSERT_EQ(slice.size(), 3u);
    EXPECT_EQ(slice[0].image_path.string(), "img_t2_c0.jpg");
    EXPECT_EQ(slice[2].image_path.string(), "img_t2_c2.jpg");
}

TEST(SequenceDatasetTest, GetTimeSliceOutOfRangeThrows) {
    auto ds = make_sequence_dataset(1, 2);
    EXPECT_THROW(ds->get_time_slice(5), std::out_of_range);
}

TEST(SequenceDatasetTest, GetFrame) {
    auto ds = make_sequence_dataset(3, 5);
    const auto& f = ds->get_frame(2, 3);
    EXPECT_EQ(f.image_path.string(), "img_t3_c2.jpg");
}

TEST(SequenceDatasetTest, GetFrameOutOfRangeThrows) {
    auto ds = make_sequence_dataset(2, 2);
    EXPECT_THROW(ds->get_frame(0, 5), std::out_of_range);
    EXPECT_THROW(ds->get_frame(5, 0), std::out_of_range);
}

TEST(SequenceDatasetTest, GetTimeStepForTimeExactMatch) {
    auto ds = make_sequence_dataset(1, 4); // ts: 0.0, 0.5, 1.0, 1.5
    EXPECT_EQ(ds->get_time_step_for_time(0.0f), 0u);
    EXPECT_EQ(ds->get_time_step_for_time(0.5f), 1u);
    EXPECT_EQ(ds->get_time_step_for_time(1.5f), 3u);
}

TEST(SequenceDatasetTest, GetTimeStepForTimeNearestNeighbour) {
    auto ds = make_sequence_dataset(1, 4);             // ts: 0.0, 0.5, 1.0, 1.5
    EXPECT_EQ(ds->get_time_step_for_time(0.24f), 0u);  // closer to 0.0
    EXPECT_EQ(ds->get_time_step_for_time(0.26f), 1u);  // closer to 0.5
    EXPECT_EQ(ds->get_time_step_for_time(100.0f), 3u); // beyond end → last
}

TEST(SequenceDatasetTest, GetTimeStepForTimeEmptyDataset) {
    std::vector<std::shared_ptr<lfs::core::Camera>> cameras;
    std::vector<float> ts;
    std::vector<std::vector<SequenceFrameEntry>> frames;
    SequenceDataset ds(std::move(cameras), std::move(ts), std::move(frames));
    EXPECT_EQ(ds.get_time_step_for_time(1.0f), 0u); // no crash, returns 0
}

// ---------------------------------------------------------------------------
// Loader::isDatasetPath / getDatasetType integration
// ---------------------------------------------------------------------------

TEST(LoaderDetectionTest, IsDatasetPathReturnsTrueFor4DDir) {
    TmpDataset ds("detect_4d");
    ds.write_manifest("{}");
    EXPECT_TRUE(Loader::isDatasetPath(ds.root));
}

TEST(LoaderDetectionTest, GetDatasetTypeReturnsSequenceFor4DDir) {
    TmpDataset ds("detect_type");
    ds.write_manifest("{}");
    EXPECT_EQ(Loader::getDatasetType(ds.root), DatasetType::Sequence);
}
