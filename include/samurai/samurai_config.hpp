// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai
{
    static constexpr bool disable_color = true;

    template <class TValue, class TIndex>
    struct Interval;

    namespace default_config
    {
        static constexpr std::size_t max_level        = 20;
        static constexpr std::size_t ghost_width      = 1;
        static constexpr std::size_t graduation_width = 1;
        static constexpr std::size_t prediction_order = 1;

        using index_t    = signed long long int;
        using value_t    = int;
        using interval_t = Interval<value_t, index_t>;
    }
}
