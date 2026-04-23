//
// Created by Andy on 4/23/2026.
//

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include <SecUtility/Math/Constant.hpp>
#include <SecUtility/Math/ContinuedFraction.hpp>


static constexpr double Tolerance = 1e-10;


using Catch::Approx;
using SecUtility::Math::ContinuedFraction;
using SecUtility::Math::SimpleContinuedFraction;

TEST_CASE("golden ratio", "[constants]")
{
	constexpr auto phi = 1 + SimpleContinuedFraction<double>([](int) { return 1; });
	constexpr double expected = SecUtility::Math::Constant::GoldenRatio<double>;
	REQUIRE(phi == Approx(expected).epsilon(Tolerance));
}

TEST_CASE("square root of 2", "[constants]")
{
	constexpr auto s = 1 + ContinuedFraction<double>([](int) { return 1; }, [](int) { return 2; });
	REQUIRE(s == Approx(SecUtility::Math::Constant::Sqrt<double>[2]).epsilon(Tolerance));
}

TEST_CASE("square root of 3", "[constants]")
{
	constexpr auto s = 1
	                   + ContinuedFraction<double>(
	                           [](int) { return 1; }, [](const int i) { return i % 2 == 0 ? 1 : 2; }, Tolerance);
	REQUIRE(s == Approx(SecUtility::Math::Constant::Sqrt<double>[3]).epsilon(Tolerance));
}

TEST_CASE("Euler's number e", "[constants]")
{
	// https://oeis.org/A003417
	constexpr std::array a003417 = {2,  1,  2,  1,  1,  4,  1,  1,  6,  1,  1,  8,  1,  1,  10, 1,  1,  12, 1,  1,
	                                14, 1,  1,  16, 1,  1,  18, 1,  1,  20, 1,  1,  22, 1,  1,  24, 1,  1,  26, 1,
	                                1,  28, 1,  1,  30, 1,  1,  32, 1,  1,  34, 1,  1,  36, 1,  1,  38, 1,  1,  40,
	                                1,  1,  42, 1,  1,  44, 1,  1,  46, 1,  1,  48, 1,  1,  50, 1,  1,  52, 1,  1,
	                                54, 1,  1,  56, 1,  1,  58, 1,  1,  60, 1,  1,  62, 1,  1,  64, 1,  1,  66};
	constexpr auto e =
	        a003417[0] + SimpleContinuedFraction<double>([&a003417](const int i) { return a003417[i + 1]; }, Tolerance);
	REQUIRE(e == Approx(SecUtility::Math::Constant::E<double>).epsilon(Tolerance));
}

TEST_CASE("pi approximation", "[constants]")
{
	// https://oeis.org/A001203
	constexpr std::array a001203 = {3,  7, 15, 1, 292, 1,  1, 1,  2,  1, 3, 1, 14, 2, 1, 1,  2, 2, 2, 2, 1, 84, 2, 1, 1,
	                                15, 3, 13, 1, 4,   2,  6, 6,  99, 1, 2, 2, 6,  3, 5, 1,  1, 6, 8, 1, 7, 1,  2, 3, 7,
	                                1,  2, 1,  1, 12,  1,  1, 1,  3,  1, 1, 8, 1,  1, 2, 1,  6, 1, 1, 5, 2, 2,  3, 1, 2,
	                                4,  4, 16, 1, 161, 45, 1, 22, 1,  2, 2, 1, 4,  1, 2, 24, 1, 2, 1, 3, 1, 2,  1};
	constexpr auto pi =
	        a001203[0] + SimpleContinuedFraction<double>([&a001203](const int i) { return a001203[i + 1]; }, 1e-12);
	REQUIRE(pi == Approx(SecUtility::Math::Constant::Pi<double>).epsilon(1e-10));
}

TEST_CASE("seed only : converges in zero steps", "[trivial]")
{
	constexpr auto val = 42 + ContinuedFraction<double>([](int) { return 0; }, [](int) { return 1; }, Tolerance);
	REQUIRE(val == Approx(42.0).epsilon(Tolerance));
}

TEST_CASE("constant fraction : all quotients identical", "[trivial]")
{
	constexpr int c = 3;
	constexpr auto val = c + SimpleContinuedFraction<double>([=](int) { return c; }, Tolerance);
	constexpr double expected = (c + SecUtility::Math::Constant::Sqrt<double>[c * c + 4]) / 2;
	REQUIRE(val == Approx(expected).epsilon(Tolerance));
}

TEST_CASE("integer seed with a_i=0 returns seed exactly", "[trivial]")
{
	for (const double seed : {0.0, 1.0, 7.0, -3.0, 100.0})
	{
		const auto val = seed + ContinuedFraction<double>([](int) { return 0; }, [](int) { return 1; }, Tolerance);
		REQUIRE(val == Approx(seed).epsilon(Tolerance));
	}
}

TEST_CASE("max_iter exceeded throws", "[error]")
{
	CHECK(std::isnan(ContinuedFraction<double>([](int) { return 1; }, [](int) { return 1; }, 1e-12, 2)));
}

TEST_CASE("zero denominator throws", "[error]")
{
	CHECK(std::isinf(SimpleContinuedFraction<double>([](int) { return 0; })));
}
