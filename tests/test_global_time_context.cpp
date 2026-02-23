/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Milestone 0: Global time plumbing — unit tests for GlobalTimeContext.

#include "core/global_time_context.hpp"

#include <gtest/gtest.h>

using namespace lfs::vis;

TEST(GlobalTimeContext, DefaultState) {
    GlobalTimeContext gtc;
    EXPECT_FLOAT_EQ(gtc.current_time, 0.0f);
    EXPECT_FALSE(gtc.time_aware);
}

TEST(GlobalTimeContext, SetCurrentTime) {
    GlobalTimeContext gtc;
    gtc.current_time = 3.14f;
    EXPECT_FLOAT_EQ(gtc.current_time, 3.14f);
}

TEST(GlobalTimeContext, TimeAwareToggle) {
    GlobalTimeContext gtc;
    EXPECT_FALSE(gtc.time_aware);

    gtc.time_aware = true;
    EXPECT_TRUE(gtc.time_aware);

    gtc.time_aware = false;
    EXPECT_FALSE(gtc.time_aware);
}

TEST(GlobalTimeContext, IndependentInstances) {
    GlobalTimeContext a;
    GlobalTimeContext b;

    a.current_time = 1.0f;
    b.current_time = 2.0f;
    a.time_aware   = true;

    EXPECT_FLOAT_EQ(a.current_time, 1.0f);
    EXPECT_FLOAT_EQ(b.current_time, 2.0f);
    EXPECT_TRUE(a.time_aware);
    EXPECT_FALSE(b.time_aware);
}
