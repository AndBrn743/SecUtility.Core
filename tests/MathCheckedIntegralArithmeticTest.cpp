// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 Andy Brown

#include <catch2/catch_test_macros.hpp>

#include <SecUtility/Math/CheckedIntegralArithmetic.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#define DUAL_CHECK(...)                                                                                                \
	CHECK(__VA_ARGS__);                                                                                                \
	STATIC_CHECK(__VA_ARGS__)

#define DUAL_CHECK_FALSE(...)                                                                                          \
	CHECK_FALSE(__VA_ARGS__);                                                                                          \
	STATIC_CHECK_FALSE(__VA_ARGS__)

using namespace SecUtility::Math;

namespace
{
	template <typename Lhs, typename Rhs, typename = void>
	inline constexpr bool CanCheckedAdd = false;

	template <typename Lhs, typename Rhs>
	inline constexpr bool
	        CanCheckedAdd<Lhs, Rhs, std::void_t<decltype(CheckedAdd(std::declval<Lhs>(), std::declval<Rhs>()))>> = true;

	template <typename Lhs, typename Rhs, typename = void>
	inline constexpr bool CanCheckedSubtract = false;

	template <typename Lhs, typename Rhs>
	inline constexpr bool
	        CanCheckedSubtract<Lhs,
	                           Rhs,
	                           std::void_t<decltype(CheckedSubtract(std::declval<Lhs>(), std::declval<Rhs>()))>> = true;

	template <typename Lhs, typename Rhs, typename = void>
	inline constexpr bool CanCheckedMultiply = false;

	template <typename Lhs, typename Rhs>
	inline constexpr bool
	        CanCheckedMultiply<Lhs,
	                           Rhs,
	                           std::void_t<decltype(CheckedMultiply(std::declval<Lhs>(), std::declval<Rhs>()))>> = true;

	template <typename Lhs, typename Rhs, typename = void>
	inline constexpr bool CanCheckedDivide = false;

	template <typename Lhs, typename Rhs>
	inline constexpr bool
	        CanCheckedDivide<Lhs, Rhs, std::void_t<decltype(CheckedDivide(std::declval<Lhs>(), std::declval<Rhs>()))>> =
	                true;
}

TEST_CASE("Checked integral arithmetic basics")
{
	DUAL_CHECK(CanCheckedAdd<int, int>);
	DUAL_CHECK(!CanCheckedAdd<int, short>);
	DUAL_CHECK(!CanCheckedSubtract<int, short>);
	DUAL_CHECK(!CanCheckedMultiply<int, short>);
	DUAL_CHECK(!CanCheckedDivide<int, short>);
	DUAL_CHECK(!CanCheckedAdd<double, double>);
	DUAL_CHECK(std::is_same_v<decltype(CheckedAdd(1, 2)), std::optional<int>>);
	DUAL_CHECK(std::is_same_v<decltype(CheckedAdd<int>(1, 2)), std::optional<int>>);
	DUAL_CHECK(std::is_same_v<decltype(CheckedAdd<long>(short{1}, std::uint8_t{2})), std::optional<long>>);
}

TEST_CASE("CheckedCast validates integral conversions")
{
	DUAL_CHECK(CheckedCast<long long>(std::numeric_limits<int>::min()) == std::numeric_limits<int>::min());
	DUAL_CHECK(CheckedCast<std::uint8_t>(255).value() == std::uint8_t{255});
	DUAL_CHECK_FALSE(CheckedCast<std::uint8_t>(256).has_value());
	DUAL_CHECK_FALSE(CheckedCast<std::uint8_t>(-1).has_value());
	DUAL_CHECK(CheckedCast<unsigned>(0).value() == 0u);
	DUAL_CHECK(CheckedCast<int>(std::uint32_t{2147483647}).value() == std::numeric_limits<int>::max());
	DUAL_CHECK_FALSE(CheckedCast<int>(std::uint32_t{2147483648u}).has_value());
	DUAL_CHECK(CheckedCast<bool>(0).value() == false);
	DUAL_CHECK(CheckedCast<bool>(1).value() == true);
	DUAL_CHECK_FALSE(CheckedCast<bool>(2).has_value());

	DUAL_CHECK(CheckedCast<std::int8_t>(std::int16_t{-128}) == std::int8_t{-128});
	DUAL_CHECK(CheckedCast<std::int8_t>(std::int16_t{127}) == std::int8_t{127});
	DUAL_CHECK_FALSE(CheckedCast<std::int8_t>(std::int16_t{-129}).has_value());
	DUAL_CHECK_FALSE(CheckedCast<std::int8_t>(std::int16_t{128}).has_value());
	DUAL_CHECK(CheckedCast<std::uint64_t>(std::numeric_limits<std::uint64_t>::max())
	      == std::numeric_limits<std::uint64_t>::max());
}

TEST_CASE("CheckedAdd detects signed and unsigned overflow")
{
	constexpr auto max = std::numeric_limits<int>::max();
	constexpr auto min = std::numeric_limits<int>::min();
	DUAL_CHECK(CheckedAdd(20, 22).value() == 42);
	DUAL_CHECK(CheckedAdd(-20, -22).value() == -42);
	DUAL_CHECK(CheckedAdd(max, 0).value() == max);
	DUAL_CHECK_FALSE(CheckedAdd(max, 1).has_value());
	DUAL_CHECK_FALSE(CheckedAdd(min, -1).has_value());
	DUAL_CHECK(CheckedAdd(20u, 22u).value() == 42u);
	DUAL_CHECK_FALSE(CheckedAdd(std::numeric_limits<unsigned>::max(), 1u).has_value());

	DUAL_CHECK(CheckedAdd<std::int16_t>(std::int8_t{40}, std::uint8_t{2}) == 42);
	DUAL_CHECK_FALSE(CheckedAdd<std::uint8_t>(-1, 2).has_value());
	DUAL_CHECK_FALSE(CheckedAdd<std::uint8_t>(200, 100).has_value());
}

TEST_CASE("CheckedSubtract detects signed and unsigned underflow")
{
	constexpr auto max = std::numeric_limits<int>::max();
	constexpr auto min = std::numeric_limits<int>::min();
	DUAL_CHECK(CheckedSubtract(50, 8).value() == 42);
	DUAL_CHECK(CheckedSubtract(-40, 2).value() == -42);
	DUAL_CHECK(CheckedSubtract(min, 0).value() == min);
	DUAL_CHECK(CheckedSubtract(max, -0).value() == max);
	DUAL_CHECK_FALSE(CheckedSubtract(min, 1).has_value());
	DUAL_CHECK_FALSE(CheckedSubtract(max, -1).has_value());
	DUAL_CHECK(CheckedSubtract(42u, 42u).value() == 0u);
	DUAL_CHECK_FALSE(CheckedSubtract(0u, 1u).has_value());

	DUAL_CHECK(CheckedSubtract<long>(std::uint8_t{50}, std::int8_t{8}) == 42L);
	DUAL_CHECK_FALSE(CheckedSubtract<unsigned>(1, -1).has_value());
}

TEST_CASE("CheckedMultiply detects every overflow sign combination")
{
	constexpr auto max = std::numeric_limits<int>::max();
	constexpr auto min = std::numeric_limits<int>::min();
	DUAL_CHECK(CheckedMultiply(6, 7).value() == 42);
	DUAL_CHECK(CheckedMultiply(-6, 7).value() == -42);
	DUAL_CHECK(CheckedMultiply(-6, -7).value() == 42);
	DUAL_CHECK(CheckedMultiply(max, 1).value() == max);
	DUAL_CHECK(CheckedMultiply(min, 1).value() == min);
	DUAL_CHECK(CheckedMultiply(min, 0).value() == 0);
	DUAL_CHECK_FALSE(CheckedMultiply(max, 2).has_value());
	DUAL_CHECK_FALSE(CheckedMultiply(min, -1).has_value());
	DUAL_CHECK_FALSE(CheckedMultiply(min, 2).has_value());
	DUAL_CHECK_FALSE(CheckedMultiply(-2, min).has_value());
	DUAL_CHECK(CheckedMultiply(0u, std::numeric_limits<unsigned>::max()).value() == 0u);
	DUAL_CHECK_FALSE(CheckedMultiply(std::numeric_limits<unsigned>::max(), 2u).has_value());

	DUAL_CHECK(CheckedMultiply<long>(std::int8_t{6}, std::uint8_t{7}) == 42L);
	DUAL_CHECK_FALSE(CheckedMultiply<std::uint8_t>(16, 16).has_value());
	DUAL_CHECK_FALSE(CheckedMultiply<std::uint8_t>(-1, 2).has_value());
}

TEST_CASE("Portable checked arithmetic fallbacks detect overflow")
{
	constexpr auto max = std::numeric_limits<int>::max();
	constexpr auto min = std::numeric_limits<int>::min();
	constexpr auto unsignedMax = std::numeric_limits<unsigned>::max();

	DUAL_CHECK_FALSE(Detail::CheckedAddFallback(max, 1).has_value());
	DUAL_CHECK_FALSE(Detail::CheckedAddFallback(min, -1).has_value());
	DUAL_CHECK_FALSE(Detail::CheckedSubtractFallback(min, 1).has_value());
	DUAL_CHECK_FALSE(Detail::CheckedSubtractFallback(max, -1).has_value());
	DUAL_CHECK_FALSE(Detail::CheckedMultiplyFallback(unsignedMax, 2u).has_value());
	DUAL_CHECK_FALSE(Detail::CheckedMultiplyFallback(max, 2).has_value());
	DUAL_CHECK_FALSE(Detail::CheckedMultiplyFallback(2, min).has_value());
	DUAL_CHECK_FALSE(Detail::CheckedMultiplyFallback(min, 2).has_value());
	DUAL_CHECK_FALSE(Detail::CheckedMultiplyFallback(-2, min).has_value());

	DUAL_CHECK(Detail::CheckedAddFallback(20, 22).value() == 42);
	DUAL_CHECK(Detail::CheckedSubtractFallback(50, 8).value() == 42);
	DUAL_CHECK(Detail::CheckedMultiplyFallback(6, 7).value() == 42);
}

TEST_CASE("CheckedDivide handles invalid divisions")
{
	constexpr auto min = std::numeric_limits<int>::min();
	DUAL_CHECK(CheckedDivide(84, 2).value() == 42);
	DUAL_CHECK(CheckedDivide(-84, 2).value() == -42);
	DUAL_CHECK(CheckedDivide(5, 2).value() == 2);
	DUAL_CHECK_FALSE(CheckedDivide(1, 0).has_value());
	DUAL_CHECK_FALSE(CheckedDivide(1u, 0u).has_value());
	DUAL_CHECK_FALSE(CheckedDivide(min, -1).has_value());
	DUAL_CHECK(CheckedDivide(min, 1).value() == min);

	DUAL_CHECK(CheckedDivide<long>(std::uint8_t{84}, std::int8_t{2}) == 42L);
	DUAL_CHECK_FALSE(CheckedDivide<unsigned>(10, -2).has_value());
	DUAL_CHECK_FALSE(CheckedDivide<std::int8_t>(256, 2).has_value());
}

TEST_CASE("CheckedNegate handles signed boundaries and unsigned zero")
{
	DUAL_CHECK(CheckedNegate(42).value() == -42);
	DUAL_CHECK(CheckedNegate(-42).value() == 42);
	DUAL_CHECK(CheckedNegate(0).value() == 0);
	DUAL_CHECK_FALSE(CheckedNegate(std::numeric_limits<int>::min()).has_value());
	DUAL_CHECK(CheckedNegate(0u).value() == 0u);
	DUAL_CHECK_FALSE(CheckedNegate(1u).has_value());
	DUAL_CHECK(CheckedNegate(false).value() == false);
	DUAL_CHECK_FALSE(CheckedNegate(true).has_value());
}

TEST_CASE("Boolean arithmetic uses the representable boolean domain")
{
	DUAL_CHECK(CheckedAdd(false, true).value() == true);
	DUAL_CHECK_FALSE(CheckedAdd(true, true).has_value());
	DUAL_CHECK(CheckedSubtract(true, true).value() == false);
	DUAL_CHECK_FALSE(CheckedSubtract(false, true).has_value());
	DUAL_CHECK(CheckedMultiply(true, true).value() == true);
	DUAL_CHECK(CheckedMultiply(false, true).value() == false);
	DUAL_CHECK(CheckedDivide(true, true).value() == true);
	DUAL_CHECK_FALSE(CheckedDivide(true, false).has_value());
}
