/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::vis {

    // Central time context for OMG4 global time plumbing (Milestone 0).
    // Driven by SequencerController playhead; queryable by any subsystem
    // (viewport, rendering, future 4D dataset/model systems).
    // Registered in Services and updated once per frame on the main thread.
    struct GlobalTimeContext {
        float current_time = 0.0f; // Sequencer playhead position in seconds
        bool  time_aware   = false; // Enable time-aware mode (e.g. 4D dataset slicing)
    };

} // namespace lfs::vis
