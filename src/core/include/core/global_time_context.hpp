/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

namespace lfs::core {

    // Global time context driven by the SequencerController playhead.
    // Queryable by any subsystem (rendering, 4D dataset/model systems, etc.).
    // Registered in Services and updated once per frame on the main thread.
    struct GlobalTimeContext {
        float current_time = 0.0f; // Sequencer playhead position in seconds
    };

} // namespace lfs::core
