// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Misc/Prefetch.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <vector>

using namespace SecUtility;

namespace
{
    // Verifies arbitrary user types work as pointees (const void* conversion).
    struct Widget
    {
        int value;
        double data[4];
    };
}

TEST_CASE("Prefetch accepts stack pointers")
{
    int x = 42;
    Prefetch(&x);

    Widget w{};
    Prefetch(&w);

    int arr[8]{};
    Prefetch(arr);
    Prefetch(arr + 4);

    CHECK(x == 42);
}

TEST_CASE("Prefetch accepts heap pointers")
{
    const std::vector<int> v(1024, 7);
    Prefetch(v.data());
    Prefetch(v.data() + v.size() / 2);

    CHECK(v.front() == 7);
}

TEST_CASE("Prefetch tolerates nullptr")
{
    Prefetch(nullptr);
}

TEST_CASE("Prefetch issues many hints in a loop")
{
    std::array<int, 1024> arr{};
    for (std::size_t i = 0; i < arr.size(); ++i)
    {
        Prefetch(arr.data() + i);
        arr[i] = static_cast<int>(i);
    }

    CHECK(arr.back() == static_cast<int>(arr.size() - 1));
}

TEST_CASE("PrefetchStream accepts stack pointers")
{
    int x = 7;
    PrefetchStream(&x);
    CHECK(x == 7);
}

TEST_CASE("PrefetchStream accepts heap pointers")
{
    const std::vector<double> v(256, 3.14);
    PrefetchStream(v.data());
    PrefetchStream(v.data() + 100);

    CHECK(v.front() == 3.14);
}

TEST_CASE("PrefetchStream tolerates nullptr")
{
    PrefetchStream(nullptr);
}

TEST_CASE("Prefetch and PrefetchStream can be mixed")
{
    std::vector<int> v(512);
    for (std::size_t i = 0; i + 64 < v.size(); ++i)
    {
        Prefetch(v.data() + i);
        PrefetchStream(v.data() + i + 64);
        v[i] = static_cast<int>(i);
    }

    CHECK(v[100] == 100);
}

TEST_CASE("Prefetch functions are noexcept")
{
    int x = 0;
    STATIC_CHECK(noexcept(Prefetch(&x)));
    STATIC_CHECK(noexcept(PrefetchStream(&x)));
}

#if defined(SEC_IF_NOT_CONSTEVAL)
TEST_CASE("Prefetch is usable in constant expressions")
{
    constexpr int value = []
    {
        int v = 42;
        Prefetch(&v);
        PrefetchStream(&v);
        return v;
    }();

    STATIC_CHECK(value == 42);
}
#endif
