/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * @file test_splat_data_4d.cpp
 * @brief Unit tests for lfs::core::SplatData4D.
 *
 * Tests cover:
 *   - Construction (empty, full constructor)
 *   - Computed getters (get_t, get_scaling_t, get_rotation_r)
 *   - Temporal marginal computation (get_marginal_t)
 *   - Raw tensor accessors
 *   - Time duration and rot_4d flag
 *   - Serialization round-trip
 *   - Inheritance from SplatData
 */

#include "core/splat_data_4d.hpp"
#include "core/tensor.hpp"

#include <gtest/gtest.h>
#include <sstream>

using namespace lfs::core;

namespace {

    /// Helper: create a simple valid SplatData4D with N Gaussians at time 0.
    std::unique_ptr<SplatData4D> make_test_model(size_t N = 4, int sh_degree = 0) {
        // 3D parameters
        auto means = Tensor::zeros({N, 3UL}, Device::CUDA, DataType::Float32);
        auto sh0 = Tensor::zeros({N, 1UL, 3UL}, Device::CUDA, DataType::Float32);
        auto shN = Tensor{};
        auto scaling = Tensor::zeros({N, 3UL}, Device::CUDA, DataType::Float32);   // log-scale 0 → scale 1
        auto rotation = Tensor::zeros({N, 4UL}, Device::CUDA, DataType::Float32);
        auto opacity = Tensor::zeros({N, 1UL}, Device::CUDA, DataType::Float32);   // logit 0 → sigmoid 0.5

        // 4D parameters
        auto t = Tensor::zeros({N, 1UL}, Device::CUDA, DataType::Float32);
        auto scaling_t = Tensor::zeros({N, 1UL}, Device::CUDA, DataType::Float32); // log-scale 0 → scale 1
        auto rotation_r = Tensor::zeros({N, 4UL}, Device::CUDA, DataType::Float32);

        // Set rotation_r w=1 (identity quaternion)
        auto rotation_r_cpu = Tensor::zeros({N, 4UL}, Device::CPU, DataType::Float32);
        for (size_t i = 0; i < N; ++i)
            rotation_r_cpu.ptr<float>()[i * 4 + 0] = 1.0f; // w component
        rotation_r = rotation_r_cpu.cuda();

        return std::make_unique<SplatData4D>(
            sh_degree,
            std::move(means),
            std::move(sh0),
            std::move(shN),
            std::move(scaling),
            std::move(rotation),
            std::move(opacity),
            1.0f,   // scene_scale
            std::move(t),
            std::move(scaling_t),
            std::move(rotation_r),
            std::array<float, 2>{0.0f, 1.0f},
            true);
    }

} // anonymous namespace

// ---------------------------------------------------------------------------
// Construction & basic accessors
// ---------------------------------------------------------------------------

TEST(SplatData4DTest, DefaultConstruction) {
    SplatData4D m;
    EXPECT_TRUE(m.t_raw().is_valid() == false);
}

TEST(SplatData4DTest, FullConstruction) {
    auto m = make_test_model(8);
    EXPECT_EQ(m->size(), 8u);
    EXPECT_TRUE(m->t_raw().is_valid());
    EXPECT_TRUE(m->scaling_t_raw().is_valid());
    EXPECT_TRUE(m->rotation_r_raw().is_valid());
}

TEST(SplatData4DTest, InheritsFromSplatData) {
    auto m = make_test_model(4);
    // SplatData interface should be accessible
    EXPECT_EQ(m->get_active_sh_degree(), 0);
    EXPECT_FLOAT_EQ(m->get_scene_scale(), 1.0f);
    EXPECT_EQ(m->size(), 4u);
}

// ---------------------------------------------------------------------------
// Computed getters
// ---------------------------------------------------------------------------

TEST(SplatData4DTest, GetTReturnsRaw) {
    auto m = make_test_model(4);
    const auto t = m->get_t();
    ASSERT_TRUE(t.is_valid());
    EXPECT_EQ(t.shape()[0], 4);
    EXPECT_EQ(t.shape()[1], 1);
}

TEST(SplatData4DTest, GetScalingTActivated) {
    // scaling_t raw = 0 → activated = exp(0) = 1
    auto m = make_test_model(4);
    const auto st = m->get_scaling_t().cpu();
    ASSERT_TRUE(st.is_valid());
    EXPECT_EQ(st.shape()[0], 4);
    const float* ptr = st.ptr<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(ptr[i], 1.0f, 1e-5f);
    }
}

TEST(SplatData4DTest, GetRotationRNormalized) {
    auto m = make_test_model(4);
    const auto rr = m->get_rotation_r().cpu();
    ASSERT_TRUE(rr.is_valid());
    EXPECT_EQ(rr.shape()[0], 4);
    EXPECT_EQ(rr.shape()[1], 4);

    // Each row should be a unit quaternion
    const float* ptr = rr.ptr<float>();
    for (int i = 0; i < 4; ++i) {
        const float w = ptr[i * 4 + 0];
        const float x = ptr[i * 4 + 1];
        const float y = ptr[i * 4 + 2];
        const float z = ptr[i * 4 + 3];
        const float norm = std::sqrt(w*w + x*x + y*y + z*z);
        EXPECT_NEAR(norm, 1.0f, 1e-5f) << "Row " << i << " not normalized";
    }
}

// ---------------------------------------------------------------------------
// Temporal marginal
// ---------------------------------------------------------------------------

TEST(SplatData4DTest, MarginalAtExactTimestamp) {
    // t_centers = 0, timestamp = 0 → dt = 0 → weight = exp(0) = 1
    auto m = make_test_model(4);
    const auto w = m->get_marginal_t(0.0f).cpu();
    ASSERT_TRUE(w.is_valid());
    EXPECT_EQ(w.shape()[0], 4);
    const float* ptr = w.ptr<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(ptr[i], 1.0f, 1e-5f);
    }
}

TEST(SplatData4DTest, MarginalDecaysWithDistance) {
    // t_centers = 0, timestamp = 2.0 → should be < 1
    auto m = make_test_model(4);
    const auto w_far = m->get_marginal_t(2.0f).cpu();
    const auto w_near = m->get_marginal_t(0.5f).cpu();

    ASSERT_TRUE(w_far.is_valid());
    ASSERT_TRUE(w_near.is_valid());

    const float w_far_0 = w_far.ptr<float>()[0];
    const float w_near_0 = w_near.ptr<float>()[0];

    EXPECT_LT(w_far_0, 1.0f);
    EXPECT_LT(w_near_0, 1.0f);
    // Farther timestamp → lower weight
    EXPECT_LT(w_far_0, w_near_0);
}

// ---------------------------------------------------------------------------
// Time duration and rot_4d flag
// ---------------------------------------------------------------------------

TEST(SplatData4DTest, TimeDurationDefault) {
    auto m = make_test_model(4);
    const auto& dur = m->time_duration();
    EXPECT_FLOAT_EQ(dur[0], 0.0f);
    EXPECT_FLOAT_EQ(dur[1], 1.0f);
}

TEST(SplatData4DTest, SetTimeDuration) {
    auto m = make_test_model(4);
    m->set_time_duration({-0.5f, 0.5f});
    EXPECT_FLOAT_EQ(m->time_duration()[0], -0.5f);
    EXPECT_FLOAT_EQ(m->time_duration()[1], 0.5f);
}

TEST(SplatData4DTest, Rot4DFlag) {
    auto m = make_test_model(4);
    EXPECT_TRUE(m->has_rot_4d());
    m->set_rot_4d(false);
    EXPECT_FALSE(m->has_rot_4d());
}

// ---------------------------------------------------------------------------
// Move semantics
// ---------------------------------------------------------------------------

TEST(SplatData4DTest, MoveConstruction) {
    auto m1 = make_test_model(4);
    SplatData4D m2(std::move(*m1));
    EXPECT_EQ(m2.size(), 4u);
    EXPECT_TRUE(m2.t_raw().is_valid());
}

TEST(SplatData4DTest, MoveAssignment) {
    auto m1 = make_test_model(4);
    SplatData4D m2;
    m2 = std::move(*m1);
    EXPECT_EQ(m2.size(), 4u);
    EXPECT_TRUE(m2.t_raw().is_valid());
}

// ---------------------------------------------------------------------------
// Serialization round-trip
// ---------------------------------------------------------------------------

TEST(SplatData4DTest, SerializeDeserializeRoundTrip) {
    auto m1 = make_test_model(4);
    m1->set_time_duration({-0.5f, 0.5f});
    m1->set_rot_4d(false);

    // Serialize
    std::ostringstream oss;
    m1->serialize(oss);

    // Deserialize into a new object
    std::istringstream iss(oss.str());
    SplatData4D m2;
    m2.deserialize(iss);

    // Verify fields
    EXPECT_EQ(m2.size(), 4u);
    EXPECT_FLOAT_EQ(m2.time_duration()[0], -0.5f);
    EXPECT_FLOAT_EQ(m2.time_duration()[1], 0.5f);
    EXPECT_FALSE(m2.has_rot_4d());
    EXPECT_TRUE(m2.t_raw().is_valid());
    EXPECT_TRUE(m2.scaling_t_raw().is_valid());
    EXPECT_TRUE(m2.rotation_r_raw().is_valid());
}
