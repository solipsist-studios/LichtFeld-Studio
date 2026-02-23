/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/global_time_context.hpp"

#include <gtest/gtest.h>

using namespace lfs::core;

TEST(GlobalTimeContext, DefaultState) {
    GlobalTimeContext gtc;
    EXPECT_FLOAT_EQ(gtc.current_time, 0.0f);
}

TEST(GlobalTimeContext, SetCurrentTime) {
    GlobalTimeContext gtc;
    gtc.current_time = 3.14f;
    EXPECT_FLOAT_EQ(gtc.current_time, 3.14f);
}

TEST(GlobalTimeContext, IndependentInstances) {
    GlobalTimeContext a;
    GlobalTimeContext b;

    a.current_time = 1.0f;
    b.current_time = 2.0f;

    EXPECT_FLOAT_EQ(a.current_time, 1.0f);
    EXPECT_FLOAT_EQ(b.current_time, 2.0f);
}
