// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Unit tests for GlobalTimeContext (Milestone 0: OMG4/4D global time context).
//
// These tests exercise GlobalTimeContext in isolation using real
// SequencerController instances, which do not require a GPU or OpenGL context.

#include "core/global_time_context.hpp"
#include "sequencer/sequencer_controller.hpp"
#include "sequencer/keyframe.hpp"
#include <gtest/gtest.h>

namespace lfs::vis {

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    static lfs::sequencer::Keyframe makeKeyframe(float time) {
        return lfs::sequencer::Keyframe{.time = time};
    }

    // ---------------------------------------------------------------------------
    // Tests: unbound context (no controller / layout)
    // ---------------------------------------------------------------------------

    TEST(GlobalTimeContext, DefaultsWhenUnbound) {
        GlobalTimeContext ctx;
        EXPECT_FLOAT_EQ(ctx.currentTime(), 0.0f);
        EXPECT_FALSE(ctx.isPlaying());
        EXPECT_FALSE(ctx.hasTimeline());
        EXPECT_FALSE(ctx.isSequencerVisible());
        // setSequencerVisible on an unbound context must not crash.
        EXPECT_NO_THROW(ctx.setSequencerVisible(true));
    }

    // ---------------------------------------------------------------------------
    // Tests: bound to a real SequencerController (no PanelLayoutManager)
    // ---------------------------------------------------------------------------

    TEST(GlobalTimeContext, ControllerOnlyBind) {
        SequencerController controller;
        GlobalTimeContext ctx;
        ctx.bind(&controller, nullptr);

        // Empty timeline -> time is 0, no playing, no timeline
        EXPECT_FLOAT_EQ(ctx.currentTime(), 0.0f);
        EXPECT_FALSE(ctx.isPlaying());
        EXPECT_FALSE(ctx.hasTimeline());
    }

    TEST(GlobalTimeContext, ReflectsPlayheadAfterSeek) {
        SequencerController controller;
        controller.addKeyframe(makeKeyframe(0.0f));
        controller.addKeyframe(makeKeyframe(5.0f));

        GlobalTimeContext ctx;
        ctx.bind(&controller, nullptr);

        EXPECT_TRUE(ctx.hasTimeline());
        controller.seek(2.5f);
        EXPECT_FLOAT_EQ(ctx.currentTime(), 2.5f);
    }

    TEST(GlobalTimeContext, ReflectsPlayingState) {
        SequencerController controller;
        controller.addKeyframe(makeKeyframe(0.0f));
        controller.addKeyframe(makeKeyframe(10.0f));

        GlobalTimeContext ctx;
        ctx.bind(&controller, nullptr);

        EXPECT_FALSE(ctx.isPlaying());
        controller.play();
        EXPECT_TRUE(ctx.isPlaying());
        controller.pause();
        EXPECT_FALSE(ctx.isPlaying());
    }

    TEST(GlobalTimeContext, HasTimelineReturnsFalseWhenEmpty) {
        SequencerController controller;
        GlobalTimeContext ctx;
        ctx.bind(&controller, nullptr);
        EXPECT_FALSE(ctx.hasTimeline());

        controller.addKeyframe(makeKeyframe(0.0f));
        EXPECT_TRUE(ctx.hasTimeline());
    }

    TEST(GlobalTimeContext, RebindUpdatesSource) {
        SequencerController ctrl1;
        ctrl1.addKeyframe(makeKeyframe(0.0f));
        ctrl1.addKeyframe(makeKeyframe(4.0f));
        ctrl1.seek(1.0f);

        SequencerController ctrl2;
        ctrl2.addKeyframe(makeKeyframe(0.0f));
        ctrl2.addKeyframe(makeKeyframe(8.0f));
        ctrl2.seek(3.0f);

        GlobalTimeContext ctx;
        ctx.bind(&ctrl1, nullptr);
        EXPECT_FLOAT_EQ(ctx.currentTime(), 1.0f);

        ctx.bind(&ctrl2, nullptr);
        EXPECT_FLOAT_EQ(ctx.currentTime(), 3.0f);
    }

} // namespace lfs::vis
