/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_four_d_loader.cpp
 * @brief Unit and integration tests for 4D dataset loading via BlenderLoader.
 *
 * Tests cover:
 *   - 4D detection via transforms.json with camera_label + multi-image directories
 *   - Happy-path loading produces correct Loaded4DDataset
 *   - Fallback to 3D when camera_label absent or single-image directory
 *   - Mismatched time step counts across cameras → hard error
 *   - Missing image directory → hard error
 *   - camera_label-absent fallback to file_path stem (3D path)
 *   - SequenceDataset access patterns (unchanged)
 */

#include "io/loader.hpp"
#include "io/loaders/blender_loader.hpp"
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

    void write_file(const fs::path& path, const std::string& content) {
        fs::create_directories(path.parent_path());
        std::ofstream f(path, std::ios::binary);
        ASSERT_TRUE(f.is_open()) << "Cannot open " << path;
        f << content;
    }

    /// Touch an empty file at @p path (creates parent dirs).
    void touch(const fs::path& path) {
        fs::create_directories(path.parent_path());
        std::ofstream f(path);
    }

    /// A minimal 4×4 identity transform_matrix JSON string.
    constexpr const char* IDENTITY_TM =
        "[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]";

    /**
     * Build a transforms.json string for @p num_cams cameras, each with
     * @p num_times images in its directory.  Each frame entry has a
     * "camera_label" field (so 4D mode is triggered) unless
     * @p include_camera_label is false.
     *
     * Image directories follow the pattern: images/cam_NNN/
     * Filenames: frame_TTTT.png
     */
    std::string make_transforms_4d(int num_cams, int num_times,
                                   bool include_camera_label = true,
                                   float fl_x = 320.0f, float fl_y = 320.0f,
                                   int w = 640, int h = 480) {
        std::string j = "{\n";
        j += std::format("  \"fl_x\": {}, \"fl_y\": {},\n", fl_x, fl_y);
        j += std::format("  \"cx\": {}, \"cy\": {},\n", w / 2.0f, h / 2.0f);
        j += std::format("  \"w\": {}, \"h\": {},\n", w, h);
        j += "  \"frames\": [\n";

        bool first = true;
        for (int c = 0; c < num_cams; ++c) {
            if (!first) j += ",\n";
            first = false;
            j += "    {\n";
            j += std::format("      \"file_path\": \"images/cam_{:03d}/frame_0000.png\",\n", c);
            if (include_camera_label)
                j += std::format("      \"camera_label\": \"Camera_{:04d}\",\n", c);
            j += std::format("      \"transform_matrix\": {}\n", IDENTITY_TM);
            j += "    }";
        }

        j += "\n  ]\n}\n";
        return j;
    }

    /// Temporary dataset directory helper.
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

        void write_transforms(const std::string& content) {
            write_file(root / "transforms.json", content);
        }

        /// Create the image files for @p num_cams cameras × @p num_times time steps.
        void touch_images(int num_cams, int num_times) {
            for (int c = 0; c < num_cams; ++c) {
                for (int t = 0; t < num_times; ++t) {
                    touch(root / std::format("images/cam_{:03d}/frame_{:04d}.png", c, t));
                }
            }
        }

        /// Create image files for one camera only (for mismatch tests).
        void touch_images_for_cam(int cam_idx, int num_times) {
            for (int t = 0; t < num_times; ++t) {
                touch(root / std::format("images/cam_{:03d}/frame_{:04d}.png", cam_idx, t));
            }
        }
    };

} // namespace

// ---------------------------------------------------------------------------
// BlenderLoader 4D detection (canLoad)
// ---------------------------------------------------------------------------

class BlenderLoader4DCanLoadTest : public ::testing::Test {
protected:
    BlenderLoader loader;
    TmpDataset ds{"canload_4d"};
};

TEST_F(BlenderLoader4DCanLoadTest, ReturnsFalseForNonExistentPath) {
    EXPECT_FALSE(loader.canLoad(ds.root / "does_not_exist"));
}

TEST_F(BlenderLoader4DCanLoadTest, ReturnsTrueForDirectoryWithTransforms) {
    ds.write_transforms(make_transforms_4d(2, 3));
    ds.touch_images(2, 3);
    EXPECT_TRUE(loader.canLoad(ds.root));
}

TEST_F(BlenderLoader4DCanLoadTest, ReturnsTrueForTransformsFileDirect) {
    ds.write_transforms(make_transforms_4d(1, 2));
    ds.touch_images(1, 2);
    EXPECT_TRUE(loader.canLoad(ds.root / "transforms.json"));
}

// ---------------------------------------------------------------------------
// 4D happy-path loading
// ---------------------------------------------------------------------------

class BlenderLoader4DLoadTest : public ::testing::Test {
protected:
    BlenderLoader loader;
    LoadOptions opts;
};

TEST_F(BlenderLoader4DLoadTest, LoadsCorrectNumberOfCamerasAndTimesteps) {
    TmpDataset ds("load_ok");
    ds.write_transforms(make_transforms_4d(3, 5));
    ds.touch_images(3, 5);

    auto result = loader.load(ds.root, opts);
    ASSERT_TRUE(result.has_value()) << result.error().format();
    EXPECT_EQ(result->loader_used, "BlenderLoader-4D");

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

TEST_F(BlenderLoader4DLoadTest, TimestampsAreFrameIndices) {
    TmpDataset ds("ts_indices");
    ds.write_transforms(make_transforms_4d(1, 4));
    ds.touch_images(1, 4);

    auto result = loader.load(ds.root, opts);
    ASSERT_TRUE(result.has_value()) << result.error().format();

    const auto& d4d = std::get<Loaded4DDataset>(result->data);
    ASSERT_EQ(d4d.timestamps.size(), 4u);
    EXPECT_FLOAT_EQ(d4d.timestamps[0], 0.0f);
    EXPECT_FLOAT_EQ(d4d.timestamps[1], 1.0f);
    EXPECT_FLOAT_EQ(d4d.timestamps[3], 3.0f);
}

TEST_F(BlenderLoader4DLoadTest, PerFrameIntrinsicsStoredInCamera) {
    TmpDataset ds("per_frame_intr");
    // Use custom focal length via top-level
    ds.write_transforms(make_transforms_4d(2, 2, true, 500.0f, 500.0f, 1024, 768));
    ds.touch_images(2, 2);

    auto result = loader.load(ds.root, opts);
    ASSERT_TRUE(result.has_value()) << result.error().format();
    const auto& d4d = std::get<Loaded4DDataset>(result->data);
    ASSERT_EQ(d4d.cameras.size(), 2u);
    EXPECT_FLOAT_EQ(d4d.cameras[0]->focal_x(), 500.0f);
    EXPECT_FLOAT_EQ(d4d.cameras[0]->focal_y(), 500.0f);
    EXPECT_EQ(d4d.cameras[0]->camera_width(), 1024);
    EXPECT_EQ(d4d.cameras[0]->camera_height(), 768);
}

TEST_F(BlenderLoader4DLoadTest, FpsFieldProducesWarning) {
    TmpDataset ds("fps_warn");
    std::string j = make_transforms_4d(1, 2);
    // Insert fps field before the closing brace
    j.pop_back(); // remove trailing \n
    j.pop_back(); // remove closing }
    j += ",\n  \"fps\": 30.0\n}\n";
    ds.write_transforms(j);
    ds.touch_images(1, 2);

    auto result = loader.load(ds.root, opts);
    ASSERT_TRUE(result.has_value()) << result.error().format();
    EXPECT_FALSE(result->warnings.empty());
    bool found_fps = false;
    for (const auto& w : result->warnings)
        if (w.find("fps") != std::string::npos) found_fps = true;
    EXPECT_TRUE(found_fps);
}

// ---------------------------------------------------------------------------
// 4D error cases
// ---------------------------------------------------------------------------

TEST_F(BlenderLoader4DLoadTest, MismatchedTimestepCountIsHardError) {
    TmpDataset ds("mismatch");
    ds.write_transforms(make_transforms_4d(2, 3));
    // Camera 0 gets 3 images, Camera 1 gets 5 → mismatch
    ds.touch_images_for_cam(0, 3);
    ds.touch_images_for_cam(1, 5);

    auto result = loader.load(ds.root, opts);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::INVALID_DATASET);
}

TEST_F(BlenderLoader4DLoadTest, MissingImageDirectoryIsHardError) {
    TmpDataset ds("missing_dir");
    ds.write_transforms(make_transforms_4d(1, 2));
    // Intentionally do NOT create image files

    auto result = loader.load(ds.root, opts);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::MISSING_REQUIRED_FILES);
}

// ---------------------------------------------------------------------------
// Fallback to 3D: no camera_label → existing BlenderLoader/3D path
// ---------------------------------------------------------------------------

TEST_F(BlenderLoader4DLoadTest, NoCameraLabelFallsBackTo3D) {
    TmpDataset ds("fallback_3d");
    // Single image per camera dir and no camera_label → 3D path
    ds.write_transforms(make_transforms_4d(2, 1, /*include_camera_label=*/false));
    // Create single images (so is_4d_transforms returns false on dir count check)
    ds.touch_images(2, 1);

    // The BlenderLoader should attempt 3D loading. It may fail because the image
    // isn't a real image file (empty), but it should NOT return a Loaded4DDataset.
    auto result = loader.load(ds.root, opts);
    // Either succeeds as LoadedScene or fails with a non-4D error.
    // Critically, if it succeeds, it must be a LoadedScene (not Loaded4DDataset).
    if (result.has_value()) {
        EXPECT_TRUE(std::holds_alternative<LoadedScene>(result->data));
    }
    // Any error is acceptable here (real images would be needed for full 3D load).
}

// ---------------------------------------------------------------------------
// Loader::isDatasetPath / getDatasetType integration
// ---------------------------------------------------------------------------

TEST(LoaderDetection4DTest, IsDatasetPathReturnsTrueForTransformsDir) {
    TmpDataset ds("detect_4d_transforms");
    ds.write_transforms(make_transforms_4d(1, 2));
    ds.touch_images(1, 2);
    EXPECT_TRUE(Loader::isDatasetPath(ds.root));
}

TEST(LoaderDetection4DTest, GetDatasetTypeReturnsTransformsForTransformsDir) {
    TmpDataset ds("detect_type_transforms");
    ds.write_transforms(make_transforms_4d(2, 3));
    ds.touch_images(2, 3);
    // A 4D transforms.json is still a Transforms dataset at the static detection level.
    // The 4D mode is detected at load time by BlenderLoader.
    EXPECT_EQ(Loader::getDatasetType(ds.root), DatasetType::Transforms);
}

// ---------------------------------------------------------------------------
// SequenceDataset access patterns (unchanged from original tests)
// ---------------------------------------------------------------------------

static std::shared_ptr<SequenceDataset> make_sequence_dataset(size_t num_cams, size_t num_times) {
    std::vector<std::shared_ptr<lfs::core::Camera>> cameras;
    cameras.reserve(num_cams);
    for (size_t c = 0; c < num_cams; ++c)
        cameras.push_back(std::make_shared<lfs::core::Camera>());

    std::vector<float> timestamps(num_times);
    for (size_t t = 0; t < num_times; ++t)
        timestamps[t] = static_cast<float>(t) * 0.5f;

    std::vector<std::vector<SequenceFrameEntry>> frames(num_times);
    for (size_t t = 0; t < num_times; ++t) {
        frames[t].resize(num_cams);
        for (size_t c = 0; c < num_cams; ++c)
            frames[t][c].image_path = std::format("img_t{}_c{}.jpg", t, c);
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
    auto ds = make_sequence_dataset(1, 4); // ts: 0.0, 0.5, 1.0, 1.5
    EXPECT_EQ(ds->get_time_step_for_time(0.24f), 0u);
    EXPECT_EQ(ds->get_time_step_for_time(0.26f), 1u);
    EXPECT_EQ(ds->get_time_step_for_time(100.0f), 3u);
}

TEST(SequenceDatasetTest, GetTimeStepForTimeEmptyDataset) {
    std::vector<std::shared_ptr<lfs::core::Camera>> cameras;
    std::vector<float> ts;
    std::vector<std::vector<SequenceFrameEntry>> frames;
    SequenceDataset ds(std::move(cameras), std::move(ts), std::move(frames));
    EXPECT_EQ(ds.get_time_step_for_time(1.0f), 0u);
}

