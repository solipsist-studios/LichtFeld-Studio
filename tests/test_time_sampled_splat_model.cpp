/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_time_sampled_splat_model.cpp
 * @brief Unit tests for lfs::training::TimeSampledSplatModel (Milestone 2).
 *
 * Tests cover:
 *   - Construction (empty, add entries)
 *   - Monotonicity enforcement
 *   - Nearest-frame time selection (get_model_index_for_time)
 *   - Accessor methods (get_timestamp, get_entry, get_entry_for_time)
 */

#include "training/time_sampled_splat_model.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace lfs::training;

// ---------------------------------------------------------------------------
// Construction & basic accessors
// ---------------------------------------------------------------------------

TEST(TimeSampledSplatModelTest, EmptyOnConstruction) {
    TimeSampledSplatModel m;
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.size(), 0u);
}

TEST(TimeSampledSplatModelTest, AddSingleEntry) {
    TimeSampledSplatModel m;
    m.add_entry(0.0f, fs::path("frame_0.ply"));
    EXPECT_FALSE(m.empty());
    EXPECT_EQ(m.size(), 1u);
}

TEST(TimeSampledSplatModelTest, AddMultipleEntries) {
    TimeSampledSplatModel m;
    for (int i = 0; i < 5; ++i) {
        m.add_entry(static_cast<float>(i) * 0.5f, fs::path("f.ply"));
    }
    EXPECT_EQ(m.size(), 5u);
}

TEST(TimeSampledSplatModelTest, GetTimestamp) {
    TimeSampledSplatModel m;
    m.add_entry(1.5f, fs::path("a.ply"));
    EXPECT_FLOAT_EQ(m.get_timestamp(0), 1.5f);
}

TEST(TimeSampledSplatModelTest, GetTimestampOutOfRangeThrows) {
    TimeSampledSplatModel m;
    m.add_entry(0.0f, fs::path("a.ply"));
    EXPECT_THROW(m.get_timestamp(1), std::out_of_range);
}

TEST(TimeSampledSplatModelTest, GetEntryOutOfRangeThrows) {
    TimeSampledSplatModel m;
    EXPECT_THROW(m.get_entry(0), std::out_of_range);
}

TEST(TimeSampledSplatModelTest, EntriesAccessor) {
    TimeSampledSplatModel m;
    m.add_entry(0.0f, fs::path("a.ply"));
    m.add_entry(1.0f, fs::path("b.ply"));
    const auto& entries = m.entries();
    ASSERT_EQ(entries.size(), 2u);
    EXPECT_FLOAT_EQ(entries[0].timestamp, 0.0f);
    EXPECT_FLOAT_EQ(entries[1].timestamp, 1.0f);
    EXPECT_EQ(entries[0].model_path, fs::path("a.ply"));
}

// ---------------------------------------------------------------------------
// Monotonicity enforcement
// ---------------------------------------------------------------------------

TEST(TimeSampledSplatModelTest, NonMonotonicTimestampThrows) {
    TimeSampledSplatModel m;
    m.add_entry(1.0f, fs::path("a.ply"));
    EXPECT_THROW(m.add_entry(0.5f, fs::path("b.ply")), std::invalid_argument);
}

TEST(TimeSampledSplatModelTest, EqualTimestampAllowed) {
    // Equal timestamps are a degenerate but valid case.
    TimeSampledSplatModel m;
    m.add_entry(1.0f, fs::path("a.ply"));
    EXPECT_NO_THROW(m.add_entry(1.0f, fs::path("b.ply")));
    EXPECT_EQ(m.size(), 2u);
}

// ---------------------------------------------------------------------------
// Nearest-frame time selection
// ---------------------------------------------------------------------------

class TimeSampledSplatModelTimeSelectionTest : public ::testing::Test {
protected:
    TimeSampledSplatModel model; // t = 0.0, 0.5, 1.0, 1.5

    void SetUp() override {
        for (int i = 0; i < 4; ++i) {
            model.add_entry(static_cast<float>(i) * 0.5f,
                            fs::path("frame_" + std::to_string(i) + ".ply"));
        }
    }
};

TEST_F(TimeSampledSplatModelTimeSelectionTest, EmptyModelReturnsZero) {
    TimeSampledSplatModel empty;
    EXPECT_EQ(empty.get_model_index_for_time(1.0f), 0u);
}

TEST_F(TimeSampledSplatModelTimeSelectionTest, ExactMatchFirst) {
    EXPECT_EQ(model.get_model_index_for_time(0.0f), 0u);
}

TEST_F(TimeSampledSplatModelTimeSelectionTest, ExactMatchMiddle) {
    EXPECT_EQ(model.get_model_index_for_time(0.5f), 1u);
}

TEST_F(TimeSampledSplatModelTimeSelectionTest, ExactMatchLast) {
    EXPECT_EQ(model.get_model_index_for_time(1.5f), 3u);
}

TEST_F(TimeSampledSplatModelTimeSelectionTest, BeforeStartClampsToFirst) {
    EXPECT_EQ(model.get_model_index_for_time(-999.0f), 0u);
}

TEST_F(TimeSampledSplatModelTimeSelectionTest, BeyondEndClampsToLast) {
    EXPECT_EQ(model.get_model_index_for_time(999.0f), 3u);
}

TEST_F(TimeSampledSplatModelTimeSelectionTest, NearestRoundsDown) {
    // 0.24 is closer to 0.0 (d=0.24) than to 0.5 (d=0.26)
    EXPECT_EQ(model.get_model_index_for_time(0.24f), 0u);
}

TEST_F(TimeSampledSplatModelTimeSelectionTest, NearestRoundsUp) {
    // 0.26 is closer to 0.5 (d=0.24) than to 0.0 (d=0.26)
    EXPECT_EQ(model.get_model_index_for_time(0.26f), 1u);
}

TEST_F(TimeSampledSplatModelTimeSelectionTest, MidpointPicksLower) {
    // Exactly at midpoint 0.25: d_lower == d_upper → pick lower (index 0)
    EXPECT_EQ(model.get_model_index_for_time(0.25f), 0u);
}

TEST_F(TimeSampledSplatModelTimeSelectionTest, GetEntryForTime) {
    const auto& entry = model.get_entry_for_time(1.0f);
    EXPECT_FLOAT_EQ(entry.timestamp, 1.0f);
    EXPECT_EQ(entry.model_path, fs::path("frame_2.ply"));
}

// ---------------------------------------------------------------------------
// Null model pointer stored in entry
// ---------------------------------------------------------------------------

TEST(TimeSampledSplatModelTest, NullModelPointerAllowed) {
    TimeSampledSplatModel m;
    // add_entry with an explicit null model (default)
    m.add_entry(0.0f, fs::path("frame_0.ply"), nullptr);
    EXPECT_EQ(m.size(), 1u);
    EXPECT_EQ(m.get_entry(0).model, nullptr);
}
