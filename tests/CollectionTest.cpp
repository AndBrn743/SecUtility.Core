//
// Created by Andy on 6/12/2026.
//

#include <catch2/catch_test_macros.hpp>
#include <SecUtility/Collection/Array.hpp>

using namespace SecUtility;

namespace
{
    constexpr auto a1 = std::array<int, 3>{1, 2, 3};
    constexpr auto a2 = std::array<int, 2>{4, 5};
    constexpr auto a3 = std::array<int, 1>{6};
}

TEST_CASE("Concat two arrays")
{
    constexpr auto result = Concat(a1, a2);
    STATIC_REQUIRE(result.size() == 5);
    STATIC_REQUIRE(result[0] == 1);
    STATIC_REQUIRE(result[1] == 2);
    STATIC_REQUIRE(result[2] == 3);
    STATIC_REQUIRE(result[3] == 4);
    STATIC_REQUIRE(result[4] == 5);
}

TEST_CASE("Concat three arrays")
{
    constexpr auto result = Concat(a1, a2, a3);
    STATIC_REQUIRE(result.size() == 6);
	STATIC_REQUIRE(result[0] == 1);
	STATIC_REQUIRE(result[1] == 2);
	STATIC_REQUIRE(result[2] == 3);
	STATIC_REQUIRE(result[3] == 4);
	STATIC_REQUIRE(result[4] == 5);
	STATIC_REQUIRE(result[5] == 6);
}

TEST_CASE("Concat single array")
{
    constexpr auto result = Concat(a1);
    STATIC_REQUIRE(result.size() == 3);
	STATIC_REQUIRE(result[0] == 1);
	STATIC_REQUIRE(result[1] == 2);
	STATIC_REQUIRE(result[2] == 3);
}

TEST_CASE("Concat with empty arrays")
{
    constexpr auto empty = std::array<int, 0>{};
    constexpr auto result = Concat(empty, a1, empty);
    STATIC_REQUIRE(result.size() == 3);
	STATIC_REQUIRE(result[0] == 1);
	STATIC_REQUIRE(result[1] == 2);
	STATIC_REQUIRE(result[2] == 3);
}

TEST_CASE("Concat copies lvalues and moves rvalues")
{
    auto src = std::array<std::string, 2>{"hello", "world"};
    const auto result = Concat(std::array<std::string, 1>{"prefix"}, std::move(src), std::array<std::string, 1>{"suffix"});
    REQUIRE(result == std::array<std::string, 4>{"prefix", "hello", "world", "suffix"});
}

TEST_CASE("Concat preserves value type")
{
    constexpr auto floats = std::array{1.0f, 2.0f};
    constexpr auto result = Concat(floats);
    STATIC_REQUIRE(std::is_same_v<decltype(result)::value_type, float>);
}
