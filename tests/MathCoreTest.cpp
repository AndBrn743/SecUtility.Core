//
// Created by Andy on 3/6/2026.
//

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Misc/Random.hpp>

using namespace SecUtility::Math;
using Catch::Approx;

static constexpr double Pi = Constant::Pi<double>;

// ============================================================================
// Basic Math Functions
// ============================================================================

TEST_CASE("Abs function")
{
	SECTION("Runtime values")
	{
		for (int i = 0; i < 10; i++)
		{
			const auto x = SecUtility::Random::NextDouble(-10, 10);
			REQUIRE(Abs(x) == Approx(std::abs(x)));
			STATIC_CHECK(std::is_same_v<decltype(Abs(x)), decltype(std::abs(x))>);
		}
	}

	SECTION("Constexpr evaluation")
	{
		constexpr auto result = Abs(-5.0);
		STATIC_CHECK(result == 5.0);
	}

	SECTION("Edge cases")
	{
		REQUIRE(Abs(0.0) == 0.0);
		REQUIRE(Abs(-0.0) == 0.0);
		REQUIRE(std::isinf(Abs(std::numeric_limits<double>::infinity())));
		REQUIRE(std::isnan(Abs(std::numeric_limits<double>::quiet_NaN())));
	}
}

TEST_CASE("SignBit function")
{
	REQUIRE(SignBit(5.0) == std::signbit(5.0));
	REQUIRE(SignBit(-5.0) == std::signbit(-5.0));
	REQUIRE(SignBit(0.0) == std::signbit(0.0));
	REQUIRE(SignBit(-0.0) == std::signbit(-0.0));
}

TEST_CASE("CopySignToTheLeft function")
{
	REQUIRE(CopySignToTheLeft(5.0, -3.0) == Approx(std::copysign(5.0, -3.0)));
	REQUIRE(CopySignToTheLeft(-5.0, 3.0) == Approx(std::copysign(-5.0, 3.0)));
	REQUIRE(CopySignToTheLeft(3.0, -0.0) == Approx(std::copysign(3.0, -0.0)));

	STATIC_CHECK(CopySignToTheLeft(-5.0, 3.0) > 5.0 - 1e-9);
	STATIC_CHECK(CopySignToTheLeft(-5.0, 3.0) < 5.0 + 1e-9);

	STATIC_CHECK(CopySignToTheLeft(5, 3) == 5);
	STATIC_CHECK(CopySignToTheLeft(5, -3) == -5);
	STATIC_CHECK(CopySignToTheLeft(-5, 3) == 5);
	STATIC_CHECK(CopySignToTheLeft(-5, -3) == -5);
}

TEST_CASE("Rounding functions")
{
	SECTION("Ceil")
	{
		REQUIRE(Ceil(3.7) == Approx(4));
		REQUIRE(Ceil(-3.7) == Approx(-3));
		REQUIRE(Ceil(3.0) == Approx(3));

		STATIC_CHECK(std::is_same_v<decltype(Ceil(3.7)), double>);
		STATIC_CHECK(Ceil(3.7) == 4);
		STATIC_CHECK(Ceil(-3.7) == -3);
		STATIC_CHECK(Ceil(3.0) == 3);

		STATIC_CHECK(std::is_same_v<decltype(Ceil<long>(3.7)), long>);
		STATIC_CHECK(Ceil<long>(3.7) == 4L);
		STATIC_CHECK(Ceil<long>(-3.7) == -3L);
		STATIC_CHECK(Ceil<long>(3.0) == 3L);

		constexpr auto result = Ceil(2.3);
		STATIC_CHECK(result == 3.0);
	}

	SECTION("Floor")
	{
		STATIC_CHECK(Floor(3.7) == 3);
		STATIC_CHECK(Floor(-3.7) == -4);
		STATIC_CHECK(Floor(3.0) == 3);

		constexpr auto result = Floor(2.8);
		STATIC_CHECK(result == 2.0);
	}

	SECTION("Round")
	{
		REQUIRE(Round(3.7) == Approx(4));
		REQUIRE(Round(3.2) == Approx(3));
		REQUIRE(Round(-3.7) == Approx(-4));
		REQUIRE(Round(2.5) == Approx(std::round(2.5)));

		constexpr auto result = Round(2.6);
		STATIC_CHECK(result == 3.0);
	}

	SECTION("Truncate")
	{
		REQUIRE(Truncate(3.7) == Approx(std::trunc(3.7)));
		REQUIRE(Truncate(-3.7) == Approx(std::trunc(-3.7)));

		constexpr auto result = Truncate(2.9);
		STATIC_CHECK(result == 2.0);
	}

	SECTION("NearByInt")
	{
		REQUIRE(NearByInt(3.7) == Approx(std::nearbyint(3.7)));
		REQUIRE(NearByInt(3.2) == Approx(std::nearbyint(3.2)));
	}
}

// ============================================================================
// Exponential and Logarithmic Functions
// ============================================================================

TEST_CASE("Exp and Log functions")
{
	SECTION("Exp")
	{
		for (int i = 0; i < 10; i++)
		{
			const auto x = SecUtility::Random::NextDouble(-5, 5);
			REQUIRE(Exp(x) == Approx(std::exp(x)).epsilon(1e-9));
		}

		constexpr auto result = Exp(1.0);
		STATIC_CHECK(result > 2.718 && result < 2.719);
	}

	SECTION("Log")
	{
		REQUIRE(Log(1.0) == Approx(std::log(1.0)));
		REQUIRE(Log(std::exp(5.0)) == Approx(5.0).epsilon(1e-9));

		constexpr auto result = Log(1.0);
		STATIC_CHECK(result == 0.0);
	}

	SECTION("Log1p")
	{
		REQUIRE(Log1p(0.0) == Approx(std::log1p(0.0)));
		REQUIRE(Log1p(1.0) == Approx(std::log1p(1.0)));
		REQUIRE(Log1p(1e-10) == Approx(std::log1p(1e-10)));
	}

	SECTION("Log2")
	{
		REQUIRE(Log2(1.0) == Approx(std::log2(1.0)));
		REQUIRE(Log2(8.0) == Approx(std::log2(8.0)));
		REQUIRE(Log2(0.5) == Approx(std::log2(0.5)));
	}

	SECTION("Log10")
	{
		REQUIRE(Log10(1.0) == Approx(std::log10(1.0)));
		REQUIRE(Log10(10.0) == Approx(std::log10(10.0)));
		REQUIRE(Log10(100.0) == Approx(std::log10(100.0)));
	}
}

// ============================================================================
// Power Functions
// ============================================================================

TEST_CASE("Power functions")
{
	SECTION("Pow")
	{
		REQUIRE(Pow(2.0, 3.0) == Approx(std::pow(2.0, 3.0)));
		REQUIRE(Pow(3.0, 2.0) == Approx(std::pow(3.0, 2.0)));

		constexpr auto result = Pow(2.0, 3.0);
		CHECK(result == Approx(8.0));
	}

	SECTION("Sqrt")
	{
		REQUIRE(Sqrt(4.0) == Approx(std::sqrt(4.0)));
		REQUIRE(Sqrt(2.0) == Approx(std::sqrt(2.0)));
		REQUIRE(Sqrt(0.0) == Approx(std::sqrt(0.0)));

		constexpr auto result = Sqrt(4.0);
		CHECK(result == Approx(2.0));
	}

	SECTION("Hypotenuse")
	{
		REQUIRE(Hypotenuse(3.0, 4.0) == Approx(5.0));
		REQUIRE(Hypotenuse(5.0, 12.0) == Approx(13.0));
		REQUIRE(Hypotenuse(3.0, 4.0, 12.0) == Approx(13.0));

		constexpr auto result = Hypotenuse(3.0, 4.0);
		STATIC_CHECK(result == 5.0);
	}

	SECTION("Cbrt")
	{
		REQUIRE(Cbrt(8.0) == Approx(std::cbrt(8.0)));
		REQUIRE(Cbrt(-8.0) == Approx(std::cbrt(-8.0)));
		REQUIRE(Cbrt(27.0) == Approx(std::cbrt(27.0)));
	}
}

// ============================================================================
// Trigonometric Functions
// ============================================================================

TEST_CASE("Trigonometric functions")
{
	SECTION("Sin")
	{
		REQUIRE(Sin(0.0) == Approx(std::sin(0.0)));
		REQUIRE(Sin(Pi / 2) == Approx(std::sin(Pi / 2)).epsilon(1e-9));
		REQUIRE(Sin(Pi) == Approx(std::sin(Pi)).epsilon(1e-9));

		constexpr auto result = Sin(0.0);
		STATIC_CHECK(result == 0.0);
	}

	SECTION("Cos")
	{
		REQUIRE(Cos(0.0) == Approx(std::cos(0.0)));
		REQUIRE(Cos(Pi / 2) == Approx(std::cos(Pi / 2)).epsilon(1e-9));
		REQUIRE(Cos(Pi) == Approx(std::cos(Pi)).epsilon(1e-9));

		constexpr auto result = Cos(0.0);
		STATIC_CHECK(result == 1.0);
	}

	SECTION("Tan")
	{
		REQUIRE(Tan(0.0) == Approx(std::tan(0.0)));
		REQUIRE(Tan(Pi / 4) == Approx(std::tan(Pi / 4)).epsilon(1e-9));
	}

	SECTION("Inverse trig")
	{
		REQUIRE(ASin(0.0) == Approx(std::asin(0.0)));
		REQUIRE(ACos(1.0) == Approx(std::acos(1.0)));
		REQUIRE(ATan(0.0) == Approx(std::atan(0.0)));
		REQUIRE(ATan(1.0) == Approx(std::atan(1.0)).epsilon(1e-9));

		SECTION("ATan2")
		{
			REQUIRE(ATan2(0.0, 1.0) == Approx(std::atan2(0.0, 1.0)));
			REQUIRE(ATan2(1.0, 1.0) == Approx(std::atan2(1.0, 1.0)).epsilon(1e-9));
			REQUIRE(ATan2(-1.0, -1.0) == Approx(std::atan2(-1.0, -1.0)).epsilon(1e-9));
		}
	}
}

// ============================================================================
// Hyperbolic Functions
// ============================================================================

TEST_CASE("Hyperbolic functions")
{
	REQUIRE(Sinh(0.0) == Approx(std::sinh(0.0)));
	REQUIRE(Cosh(0.0) == Approx(std::cosh(0.0)));
	REQUIRE(Tanh(0.0) == Approx(std::tanh(0.0)));
	REQUIRE(ASinh(0.0) == Approx(std::asinh(0.0)));
	REQUIRE(ACosh(1.0) == Approx(std::acosh(1.0)));
	REQUIRE(ATanh(0.0) == Approx(std::atanh(0.0)));
}

// ============================================================================
// Complex Number Functions
// ============================================================================

TEST_CASE("Complex number functions")
{
	using Complex = std::complex<double>;

	SECTION("Re and Im")
	{
		const Complex c{3.0, 4.0};
		REQUIRE(Re(c) == 3.0);
		REQUIRE(Im(c) == 4.0);

		constexpr Complex cc{2.0, 5.0};
		STATIC_CHECK(Re(cc) == 2.0);
		STATIC_CHECK(Im(cc) == 5.0);
	}

	SECTION("Re and Im for real numbers")
	{
		REQUIRE(Re(5.0) == 5.0);
		REQUIRE(Im(5.0) == 0.0);

		constexpr double x = 7.0;
		STATIC_CHECK(Re(x) == 7.0);
		STATIC_CHECK(Im(x) == 0.0);
	}

	SECTION("Conj for complex")
	{
		const Complex c{3.0, 4.0};
		auto conj = Conj(c);
		REQUIRE(Re(conj) == 3.0);
		REQUIRE(Im(conj) == -4.0);

		constexpr Complex cc{2.0, -3.0};
		constexpr auto cconj = Conj(cc);
		STATIC_CHECK(Re(cconj) == 2.0);
		STATIC_CHECK(Im(cconj) == 3.0);
	}

	SECTION("Conj for real")
	{
		REQUIRE(Conj(5.0) == 5.0);

		constexpr double x = 7.0;
		STATIC_CHECK(Conj(x) == 7.0);
	}

	SECTION("Arg for complex")
	{
		const Complex c1{1.0, 0.0};
		REQUIRE(Arg(c1) == Approx(0.0));

		const Complex c2{-1.0, 0.0};
		REQUIRE(Arg(c2) == Approx(Pi));

		const Complex c3{1.0, 1.0};
		REQUIRE(Arg(c3) == Approx(Pi / 4).epsilon(1e-9));

		const Complex c4{1.0, -1.0};
		REQUIRE(Arg(c4) == Approx(-Pi / 4).epsilon(1e-9));
	}

	SECTION("Arg for real")
	{
		REQUIRE(Arg(5.0) == Approx(0.0));
		REQUIRE(Arg(-5.0) == Approx(Pi));
		REQUIRE(Arg(0.0) == Approx(0.0));
	}

	SECTION("SquaredNorm")
	{
		const Complex c{3.0, 4.0};
		REQUIRE(SquaredNorm(c) == Approx(25.0));

		constexpr double x = 5.0;
		STATIC_CHECK(SquaredNorm(x) == 25.0);
	}

	SECTION("Norm")
	{
		const Complex c{3.0, 4.0};
		REQUIRE(Norm(c) == Approx(5.0));

		constexpr Complex z{3.0, 4.0};
		STATIC_CHECK(Abs(Norm(z) - 5.0) < 1e-12);

		double y = -5.0;
		CHECK(Norm(y) == 5.0);
	}

	SECTION("Abs")
	{
		const Complex c{3.0, 4.0};
		REQUIRE(Abs(c) == Approx(5.0));


		constexpr Complex z{3.0, 4.0};
		STATIC_CHECK(Abs(Abs(z) - 5.0) < 1e-12);
	}
}

// ============================================================================
// Integer Utility Functions
// ============================================================================

TEST_CASE("Integer utility functions")
{
	SECTION("IsOdd")
	{
		REQUIRE(IsOdd(1));
		REQUIRE(IsOdd(3));
		REQUIRE(IsOdd(-1));
		REQUIRE(IsOdd(-3));
		REQUIRE(!IsOdd(0));
		REQUIRE(!IsOdd(2));
		REQUIRE(!IsOdd(-2));

		STATIC_CHECK(IsOdd(5));
		STATIC_CHECK(!IsOdd(4));
	}

	SECTION("IsEven")
	{
		REQUIRE(IsEven(0));
		REQUIRE(IsEven(2));
		REQUIRE(IsEven(-2));
		REQUIRE(!IsEven(1));
		REQUIRE(!IsEven(-1));

		STATIC_CHECK(IsEven(4));
		STATIC_CHECK(!IsEven(5));
	}

	SECTION("IsPowerOfTwo")
	{
		REQUIRE(IsPowerOfTwo(1));
		REQUIRE(IsPowerOfTwo(2));
		REQUIRE(IsPowerOfTwo(4));
		REQUIRE(IsPowerOfTwo(8));
		REQUIRE(IsPowerOfTwo(16));
		REQUIRE(IsPowerOfTwo(1024));
		REQUIRE(!IsPowerOfTwo(0));
		REQUIRE(!IsPowerOfTwo(3));
		REQUIRE(!IsPowerOfTwo(5));
		REQUIRE(!IsPowerOfTwo(6));
		REQUIRE(!IsPowerOfTwo(7));
		REQUIRE(!IsPowerOfTwo(-2));
		REQUIRE(!IsPowerOfTwo(-8));

		STATIC_CHECK(IsPowerOfTwo(8));
		STATIC_CHECK(!IsPowerOfTwo(7));
	}
}

// ============================================================================
// Comparison Functions
// ============================================================================

TEST_CASE("Comparison functions")
{
	SECTION("Max")
	{
		REQUIRE(Max(1, 2) == 2);
		REQUIRE(Max(2, 1) == 2);
		REQUIRE(Max(1.0, 2.0, 3.0) == 3.0);
		REQUIRE(Max(3.0, 2.0, 1.0) == 3.0);
		REQUIRE(Max(5, 1, 3, 9, 2) == 9);

		constexpr auto result = Max(1, 2, 3);
		STATIC_CHECK(result == 3);

		int a = 3;
		int b = 5;
		int c = 42;
		int d = -1;
		int e = 27;
		STATIC_CHECK(std::is_same_v<decltype(Max(a)), int&>);
		STATIC_CHECK(std::is_same_v<decltype(Max(a, b, c)), int&>);
		STATIC_CHECK(std::is_same_v<decltype(Max(a, b, 3)), int>);
		STATIC_CHECK(std::is_same_v<decltype(Max(3)), int&&>);
		Max(a, b, c, d, e) *= 10;
		CHECK(a == 3);
		CHECK(b == 5);
		CHECK(c == 420);
		CHECK(d == -1);
		CHECK(e == 27);

		Max(a) += 10;
		CHECK(a == 13);
	}

	SECTION("Min")
	{
		REQUIRE(Min(1, 2) == 1);
		REQUIRE(Min(2, 1) == 1);
		REQUIRE(Min(1.0, 2.0, 3.0) == 1.0);
		REQUIRE(Min(3.0, 2.0, 1.0) == 1.0);
		REQUIRE(Min(5, 1, 3, 9, 2) == 1);

		constexpr auto result = Min(1, 2, 3);
		STATIC_CHECK(result == 1);


		int a = 3;
		int b = 5;
		int c = 42;
		int d = -1;
		int e = 27;
		STATIC_CHECK(std::is_same_v<decltype(Min(a)), int&>);
		STATIC_CHECK(std::is_same_v<decltype(Min(a, b, c)), int&>);
		STATIC_CHECK(std::is_same_v<decltype(Min(a, b, 3)), int>);
		STATIC_CHECK(std::is_same_v<decltype(Min(3)), int&&>);
		Min(a, b, c, d, e) *= 10;
		CHECK(a == 3);
		CHECK(b == 5);
		CHECK(c == 42);
		CHECK(d == -10);
		CHECK(e == 27);

		Min(a) += 10;
		CHECK(a == 13);
	}

	SECTION("Extrema with custom comparator")
	{
		REQUIRE(Extrema<std::greater<>>(1, 2, 3) == 3);
		REQUIRE(Extrema<std::less<>>(1, 2, 3) == 1);
	}
}

// ============================================================================
// Accumulation Functions
// ============================================================================

TEST_CASE("Accumulation functions")
{
	SECTION("Sum")
	{
		REQUIRE(Sum(1, 2, 3, 4, 5) == 15);
		REQUIRE(Sum(1.0, 2.0, 3.0) == 6.0);
		REQUIRE(Sum(1) == 1);

		constexpr auto result = Sum(1, 2, 3, 4);
		STATIC_CHECK(result == 10);
	}

	SECTION("Prod")
	{
		REQUIRE(Prod(1, 2, 3, 4) == 24);
		REQUIRE(Prod(2.0, 3.0, 4.0) == 24.0);
		REQUIRE(Prod(5) == 5);

		constexpr auto result = Prod(2, 3, 4);
		STATIC_CHECK(result == 24);
	}
}

// ============================================================================
// Identity Function
// ============================================================================

TEST_CASE("Identity function")
{
	SECTION("Returns input unchanged")
	{
		int x = 5;
		REQUIRE(Identity(x) == 5);
		REQUIRE(&Identity(x) == &x);

		const int y = 10;
		REQUIRE(Identity(y) == 10);
		REQUIRE(&Identity(y) == &y);
	}

	SECTION("With rvalue")
	{
		REQUIRE(Identity(42) == 42);
		REQUIRE(Identity(std::string("test")) == "test");
	}

	SECTION("Constexpr")
	{
		constexpr auto result = Identity(42);
		STATIC_CHECK(result == 42);
	}
}

// ============================================================================
// New Functions (assuming they've been added)
// ============================================================================

TEST_CASE("FusedMultiplyAdd")
{
	SECTION("Basic functionality")
	{
		REQUIRE(FusedMultiplyAdd(2.0, 3.0, 4.0) == Approx(std::fma(2.0, 3.0, 4.0)));
		REQUIRE(FusedMultiplyAdd(1.5, 2.5, 3.5) == Approx(std::fma(1.5, 2.5, 3.5)));
	}

	SECTION("Constexpr")
	{
		constexpr auto result = FusedMultiplyAdd(2.0, 3.0, 1.0);
		STATIC_CHECK(result == 7.0);
	}
}

TEST_CASE("Clamp function")
{
	SECTION("Within range")
	{
		REQUIRE(Clamp(5, 0, 10) == 5);
		REQUIRE(Clamp(5.0, 0.0, 10.0) == 5.0);
	}

	SECTION("Below minimum")
	{
		REQUIRE(Clamp(-5, 0, 10) == 0);
		REQUIRE(Clamp(-5.0, 0.0, 10.0) == 0.0);
	}

	SECTION("Above maximum")
	{
		REQUIRE(Clamp(15, 0, 10) == 10);
		REQUIRE(Clamp(15.0, 0.0, 10.0) == 10.0);
	}

	SECTION("At boundaries")
	{
		REQUIRE(Clamp(0, 0, 10) == 0);
		REQUIRE(Clamp(10, 0, 10) == 10);
		REQUIRE(Clamp(0.0, 0.0, 10.0) == 0.0);
		REQUIRE(Clamp(10.0, 0.0, 10.0) == 10.0);
	}

	SECTION("Constexpr")
	{
		constexpr auto result = Clamp(15, 0, 10);
		STATIC_CHECK(result == 10);
	}
}

TEST_CASE("LinearInterpolation")
{
	SECTION("Basic interpolation")
	{
		REQUIRE(LinearInterpolation(0.0, 10.0, 0.5) == Approx(5.0));
		REQUIRE(LinearInterpolation(0.0, 10.0, 0.0) == Approx(0.0));
		REQUIRE(LinearInterpolation(0.0, 10.0, 1.0) == Approx(10.0));
		REQUIRE(LinearInterpolation(5.0, 15.0, 0.25) == Approx(7.5));
	}

	SECTION("Extrapolation")
	{
		REQUIRE(LinearInterpolation(10.0, 0.0, -0.5) == Approx(10));
		REQUIRE(LinearInterpolation(0.0, 10.0, 1.5) == Approx(10));
	}

	SECTION("Integer types")
	{
		STATIC_CHECK(LinearInterpolation(0, 10, 0.5) == 5);
		STATIC_CHECK(LinearInterpolation(0, 100, 0.25) == 25);
	}

	SECTION("Constexpr")
	{
		constexpr auto result = LinearInterpolation(0.0, 10.0, 0.5);
		STATIC_CHECK(result == 5.0);
	}
}

TEST_CASE("Sign function (returns -1, 0, 1)")
{
	SECTION("Positive numbers")
	{
		REQUIRE(Sign(5) == 1);
		REQUIRE(Sign(5.0) == 1);
		REQUIRE(Sign(0.5) == 1);
		REQUIRE(Sign(1000) == 1);
	}

	SECTION("Negative numbers")
	{
		REQUIRE(Sign(-5) == -1);
		REQUIRE(Sign(-5.0) == -1);
		REQUIRE(Sign(-0.5) == -1);
		REQUIRE(Sign(-1000) == -1);
	}

	SECTION("Zero")
	{
		REQUIRE(Sign(0) == 0);
		REQUIRE(Sign(0.0) == 0);
		REQUIRE(Sign(-0.0) == 0);
	}

	SECTION("Constexpr")
	{
		constexpr auto pos = Sign(42);
		constexpr auto zero = Sign(0);
		constexpr auto neg = Sign(-42);

		STATIC_CHECK(pos == 1);
		STATIC_CHECK(zero == 0);
		STATIC_CHECK(neg == -1);
	}
}

// TEST_CASE("Fmod (floating point remainder)")
// {
// 	SECTION("Basic functionality")
// 	{
// 		REQUIRE(Fmod(5.7, 2.0) == Approx(std::fmod(5.7, 2.0)));
// 		REQUIRE(Fmod(-5.7, 2.0) == Approx(std::fmod(-5.7, 2.0)));
// 		REQUIRE(Fmod(5.7, -2.0) == Approx(std::fmod(5.7, -2.0)));
// 	}
//
// 	SECTION("Special cases")
// 	{
// 		REQUIRE(Fmod(0.0, 1.0) == Approx(std::fmod(0.0, 1.0)));
// 		REQUIRE(Fmod(5.0, 5.0) == Approx(std::fmod(5.0, 5.0)));
// 	}
// }
//
// TEST_CASE("Remainder function (IEEE remainder)")
// {
// 	SECTION("Basic functionality")
// 	{
// 		REQUIRE(Remainder(5.7, 2.0) == Approx(std::remainder(5.7, 2.0)));
// 		REQUIRE(Remainder(-5.7, 2.0) == Approx(std::remainder(-5.7, 2.0)));
// 		REQUIRE(Remainder(5.7, -2.0) == Approx(std::remainder(5.7, -2.0)));
// 	}
//
// 	SECTION("Difference from fmod")
// 	{
// 		// remainder can return negative results
// 		auto rem1 = Fmod(7.0, 4.0);
// 		auto rem2 = Remainder(7.0, 4.0);
// 		// They may differ in sign or magnitude
// 		REQUIRE(std::abs(rem1) <= 4.0);
// 		REQUIRE(std::abs(rem2) <= 2.0);
// 	}
// }
//
// TEST_CASE("Saturate function")
// {
// 	SECTION("Basic functionality")
// 	{
// 		REQUIRE(Saturate(0.5) == 0.5);
// 		REQUIRE(Saturate(0.0) == 0.0);
// 		REQUIRE(Saturate(1.0) == 1.0);
// 		REQUIRE(Saturate(-0.5) == 0.0);
// 		REQUIRE(Saturate(1.5) == 1.0);
// 		REQUIRE(Saturate(-1.0) == 0.0);
// 		REQUIRE(Saturate(2.0) == 1.0);
// 	}
//
// 	SECTION("Constexpr")
// 	{
// 		constexpr auto result = Saturate(1.5);
// 		STATIC_CHECK(result == 1.0);
// 	}
// }

TEST_CASE("Degree/Radian conversion")
{
	SECTION("DegToRad")
	{
		REQUIRE(Deg2Rad(0.0) == Approx(0.0));
		REQUIRE(Deg2Rad(90.0) == Approx(Pi / 2).epsilon(1e-9));
		REQUIRE(Deg2Rad(180.0) == Approx(Pi).epsilon(1e-9));
		REQUIRE(Deg2Rad(360.0) == Approx(2 * Pi).epsilon(1e-9));
		REQUIRE(Deg2Rad(-90.0) == Approx(-Pi / 2).epsilon(1e-9));

		constexpr auto result = Deg2Rad(180.0);
		STATIC_CHECK(result > 3.1415 && result < 3.1416);
	}

	SECTION("RadToDeg")
	{
		REQUIRE(Rad2Deg(0.0) == Approx(0.0));
		REQUIRE(Rad2Deg(Pi / 2) == Approx(90.0).epsilon(1e-9));
		REQUIRE(Rad2Deg(Pi) == Approx(180.0).epsilon(1e-9));
		REQUIRE(Rad2Deg(2 * Pi) == Approx(360.0).epsilon(1e-9));

		constexpr auto result = Rad2Deg(Pi);
		STATIC_CHECK(result > 179.9 && result < 180.1);
	}

	SECTION("Roundtrip conversion")
	{
		constexpr double original = 45.0;
		REQUIRE(Rad2Deg(Deg2Rad(original)) == Approx(original).epsilon(1e-9));

		constexpr double rad = Pi / 3;
		REQUIRE(Deg2Rad(Rad2Deg(rad)) == Approx(rad).epsilon(1e-9));
	}
}

// TEST_CASE("Floating point classification functions")
// {
// 	SECTION("IsFinite")
// 	{
// 		REQUIRE(IsFinite(0.0));
// 		REQUIRE(IsFinite(1.0));
// 		REQUIRE(IsFinite(-1.0));
// 		REQUIRE(IsFinite(std::numeric_limits<double>::max()));
// 		REQUIRE(IsFinite(std::numeric_limits<double>::lowest()));
// 		REQUIRE(!IsFinite(std::numeric_limits<double>::infinity()));
// 		REQUIRE(!IsFinite(std::numeric_limits<double>::quiet_NaN()));
// 	}
//
// 	SECTION("IsInf")
// 	{
// 		REQUIRE(!IsInf(0.0));
// 		REQUIRE(!IsInf(1.0));
// 		REQUIRE(IsInf(std::numeric_limits<double>::infinity()));
// 		REQUIRE(IsInf(-std::numeric_limits<double>::infinity()));
// 		REQUIRE(!IsInf(std::numeric_limits<double>::quiet_NaN()));
// 	}
//
// 	SECTION("IsNan")
// 	{
// 		REQUIRE(!IsNan(0.0));
// 		REQUIRE(!IsNan(1.0));
// 		REQUIRE(!IsNan(std::numeric_limits<double>::infinity()));
// 		REQUIRE(IsNan(std::numeric_limits<double>::quiet_NaN()));
// 		REQUIRE(IsNan(std::numeric_limits<double>::signaling_NaN()));
// 	}
//
// 	SECTION("IsNormal")
// 	{
// 		REQUIRE(IsNormal(1.0));
// 		REQUIRE(IsNormal(-1.0));
// 		REQUIRE(IsNormal(std::numeric_limits<double>::min()));
//
// 		// Subnormal numbers
// 		const double subnormal = std::numeric_limits<double>::min() / 2;
// 		REQUIRE(!IsNormal(subnormal));
// 		REQUIRE(!IsNormal(0.0));
// 		REQUIRE(!IsNormal(std::numeric_limits<double>::infinity()));
// 		REQUIRE(!IsNormal(std::numeric_limits<double>::quiet_NaN()));
// 	}
//
// 	SECTION("Constexpr")
// 	{
// 		STATIC_CHECK(IsFinite(1.0));
// 		STATIC_CHECK(!IsInf(1.0));
// 		STATIC_CHECK(!IsNan(1.0));
// 	}
// }

// ============================================================================
// Mean Functions
// ============================================================================

TEST_CASE("ArithmeticMean")
{
	SECTION("Basic functionality")
	{
		STATIC_CHECK(std::is_same_v<decltype(ArithmeticMean(1, 2, 3, 4, 5)), int>);
		REQUIRE(ArithmeticMean(1, 2, 3, 4, 5) == 3);
		STATIC_CHECK(std::is_same_v<decltype(ArithmeticMean(10.0, 20.0)), double>);
		STATIC_CHECK(std::is_same_v<decltype(ArithmeticMean(10.f, 20.0)), double>);
		STATIC_CHECK(std::is_same_v<decltype(ArithmeticMean(10.0f, 20.0f)), float>);
		REQUIRE(ArithmeticMean(10.0, 20.0) == 15.0);
		REQUIRE(ArithmeticMean(5) == 5);
		REQUIRE(ArithmeticMean(0.0, 10.0) == 5.0);
	}

	SECTION("Negative numbers")
	{
		REQUIRE(ArithmeticMean(-10, 10) == 0);
		REQUIRE(ArithmeticMean(-5, -15) == -10);
		REQUIRE(ArithmeticMean(-1, 0, 1) == 0);
	}

	SECTION("Mixed types")
	{
		REQUIRE(ArithmeticMean(1, 2.0, 3) == Approx(2.0));
		REQUIRE(ArithmeticMean(1.0f, 2.0, 3.0) == Approx(2.0));
	}

	SECTION("Constexpr")
	{
		constexpr auto result = ArithmeticMean(1., 2, 3, 4);
		STATIC_CHECK(result == 2.5);
		constexpr auto integer = ArithmeticMean(1, 2, 3, 4);
		STATIC_CHECK(integer == 2);
		constexpr auto single = ArithmeticMean(42);
		STATIC_CHECK(single == 42);
	}

	SECTION("Large numbers")
	{
		REQUIRE(ArithmeticMean(1000000, 2000000, 3000000) == 2000000);
		REQUIRE(ArithmeticMean(1e10, 2e10, 3e10) == Approx(2e10));
	}
}

TEST_CASE("QuadraticMean (RMS)")
{
	SECTION("Basic functionality")
	{
		REQUIRE(QuadraticMean(3, 4) == Approx(Sqrt((3. * 3 + 4 * 4) / 2)));
		REQUIRE(QuadraticMean(1, 1, 1, 1) == Approx(1.0));
		REQUIRE(QuadraticMean(0, 0) == Approx(0.0));
	}

	SECTION("Single value")
	{
		REQUIRE(QuadraticMean(5) == Approx(5.0));
		REQUIRE(QuadraticMean(3.0) == Approx(3.0));
	}

	SECTION("Constexpr")
	{
		constexpr auto result = QuadraticMean(3.0, 4.0);
		STATIC_CHECK(result > 3.535533);
		STATIC_CHECK(result < 3.535534);
	}

	SECTION("Physical interpretation")
	{
		// RMS of AC signal values
		REQUIRE(QuadraticMean(1.0, -1.0) == Approx(1.0));
		REQUIRE(QuadraticMean(2.0, -2.0, 2.0, -2.0) == Approx(2.0));
	}
}

TEST_CASE("GeometricMean")
{
	SECTION("Basic functionality")
	{
		REQUIRE(GeometricMean(1, 4) == Approx(2.0));  // sqrt(1*4) = 2
		REQUIRE(GeometricMean(2, 8) == Approx(4.0));  // sqrt(2*8) = 4
		REQUIRE(GeometricMean(1., 2, 4, 8) == Approx(Sqrt(Sqrt(1 * 2 * 4 * 8))));
	}

	SECTION("Single value")
	{
		REQUIRE(GeometricMean(5) == 5);
		REQUIRE(GeometricMean(3.0) == Approx(3.0));
		REQUIRE(GeometricMean(1) == 1);
	}

	SECTION("Two values (square root)")
	{
		REQUIRE(GeometricMean(1, 9) == Approx(3.0));
		REQUIRE(GeometricMean(16, 25) == Approx(20.0));
	}

	SECTION("Three values (cube root)")
	{
		REQUIRE(GeometricMean(1, 8, 27) == Approx(6.0));  // cbrt(216) = 6
		REQUIRE(GeometricMean(2, 4, 8) == Approx(4.0));   // cbrt(64) = 4
	}

	SECTION("Many values (power)")
	{
		REQUIRE(GeometricMean(1., 2, 3, 4) == Approx(Pow(24, 1.0 / 4.0)));
		REQUIRE(GeometricMean(2., 4, 8, 16, 32) == Approx(Pow(2 * 4 * 8 * 16 * 32, 1.0 / 5.0)));
	}

	SECTION("Constexpr")
	{
		constexpr auto twoValues = GeometricMean(1.0, 4.0);
		STATIC_CHECK(twoValues > 1.9 && twoValues < 2.1);

		constexpr auto threeValues = GeometricMean(1.0, 8.0, 27.0);
		STATIC_CHECK(threeValues > 5.9 && threeValues < 6.1);

		constexpr auto single = GeometricMean(42);
		STATIC_CHECK(single == 42);
	}

	SECTION("Special cases")
	{
		REQUIRE(GeometricMean(1, 1, 1) == Approx(1.0));
		REQUIRE(GeometricMean(0.0, 0.0) == Approx(0.0));
	}
}

TEST_CASE("HarmonicMean")
{
	SECTION("Basic functionality")
	{
		REQUIRE(HarmonicMean(1., 1) == Approx(1));
		REQUIRE(HarmonicMean(1., 2) == Approx(2 * 2. / 3.));  // 2/(1+0.5) = 4/3
		REQUIRE(HarmonicMean(2., 4) == Approx(2 * 8. / 6.));  // 2/(0.5+0.25) = 8/3
	}

	SECTION("Speed averaging")
	{
		REQUIRE(HarmonicMean(60.0, 120.0) == Approx(2 * 60. * 120. / (60. + 120.)));
	}

	SECTION("Multiple values")
	{
		STATIC_CHECK(std::is_same_v<decltype(HarmonicMean(1, 2, 4)), double>);
		REQUIRE(HarmonicMean(1, 2, 4) == Approx(12. / 7.));
		REQUIRE(HarmonicMean(2, 3, 6) == Approx(3));
	}

	SECTION("Single value")
	{
		REQUIRE(HarmonicMean(5) == 5);
		REQUIRE(HarmonicMean(3.0) == Approx(3.0));
	}

	SECTION("Constexpr")
	{
		constexpr auto result = HarmonicMean(2.0, 3.0);
		CHECK(result > 2.4 - 1e-9);
		CHECK(result < 2.4 + 1e-9);
		constexpr auto single = HarmonicMean(42);
		STATIC_CHECK(single == 42);
	}

	SECTION("All same values")
	{
		REQUIRE(HarmonicMean(5, 5, 5) == Approx(5.0));
		REQUIRE(HarmonicMean(10, 10) == Approx(10.0));
	}
}

TEST_CASE("Mean functions edge cases")
{
	SECTION("Large value ranges")
	{
		REQUIRE(ArithmeticMean(1e-10, 1e10) > 1e9);
		REQUIRE(GeometricMean(1e-10, 1e10) == Approx(1.0));
		REQUIRE(HarmonicMean(1e-10, 1e10) < 1.0);
	}

	SECTION("Constexpr compilation smoke test")
	{
		// Ensure these compile and evaluate at compile time
		STATIC_CHECK(ArithmeticMean(1, 2, 3) == 2);
		STATIC_CHECK(ArithmeticMean(10.0, 20.0) == 15.0);

		constexpr auto qm = QuadraticMean(3.0, 4.0);
		STATIC_CHECK(qm > 3.535533);
		STATIC_CHECK(qm < 3.535534);

		STATIC_CHECK(GeometricMean(4) == 4);
		STATIC_CHECK(GeometricMean(1, 4) > 1.0 && GeometricMean(1, 4) < 3.0);

		STATIC_CHECK(HarmonicMean(5) == 5);
	}
}

// ============================================================================
// PowInt Function
// ============================================================================

TEST_CASE("PowInt function")
{
	SECTION("Basic positive exponents")
	{
		REQUIRE(PowInt(2.0, 3) == Approx(8.0));
		REQUIRE(PowInt(3.0, 4) == Approx(81.0));
		REQUIRE(PowInt(5.0, 2) == Approx(25.0));
		REQUIRE(PowInt(1.5, 3) == Approx(3.375));
		REQUIRE(PowInt(2.0, 10) == Approx(1024.0));
	}

	SECTION("Zero exponent")
	{
		REQUIRE(PowInt(2.0, 0) == 1.0);
		REQUIRE(PowInt(0.0, 0) == 1.0);
		REQUIRE(PowInt(-5.0, 0) == 1.0);
		REQUIRE(PowInt(100.0, 0) == 1.0);

		constexpr auto result_ct = PowInt(5.0, 0);
		const auto result_rt = PowInt(5.0, 0);  // NOLINT
		STATIC_CHECK(result_ct == 1.0);
		CHECK(result_rt == 1.0);
	}

	SECTION("Exponent of 1")
	{
		REQUIRE(PowInt(2.0, 1) == Approx(2.0));
		REQUIRE(PowInt(-3.0, 1) == Approx(-3.0));
		REQUIRE(PowInt(0.0, 1) == Approx(0.0));
	}

	SECTION("Negative exponents")
	{
		REQUIRE(PowInt(2.0, -1) == Approx(0.5));
		REQUIRE(PowInt(2.0, -2) == Approx(0.25));
		REQUIRE(PowInt(10.0, -3) == Approx(0.001));
		REQUIRE(PowInt(3.0, -2) == Approx(1.0 / 9.0));
	}

	SECTION("Negative bases")
	{
		REQUIRE(PowInt(-2.0, 2) == Approx(4.0));
		REQUIRE(PowInt(-2.0, 3) == Approx(-8.0));
		REQUIRE(PowInt(-3.0, 4) == Approx(81.0));
		REQUIRE(PowInt(-1.5, 3) == Approx(-3.375));
	}

	SECTION("Fractional bases")
	{
		REQUIRE(PowInt(0.5, 2) == Approx(0.25));
		REQUIRE(PowInt(1.5, 3) == Approx(3.375));
		REQUIRE(PowInt(2.5, 4) == Approx(39.0625));
	}

	SECTION("Large exponents (uses std::pow)")
	{
		REQUIRE(PowInt(2.0, 40) == Approx(std::pow(2.0, 40)));
		REQUIRE(PowInt(1.5, 50) == Approx(std::pow(1.5, 50)));
		REQUIRE(PowInt(2.0, -40) == Approx(std::pow(2.0, -40)));
	}

	SECTION("Constexpr evaluation")
	{
		constexpr auto result1 = PowInt(2.0, 10);
		STATIC_CHECK(result1 == 1024.0);

		constexpr auto result2 = PowInt(3.0, 5);
		STATIC_CHECK(result2 == 243.0);

		constexpr auto result3 = PowInt(2.0, -3);
		STATIC_CHECK(result3 > 0.124 && result3 < 0.126);

		// Runtime version
		const auto result_rt = PowInt(5.0, 3);  // NOLINT
		CHECK(result_rt == Approx(125.0));
	}

	SECTION("Different scalar types")
	{
		STATIC_CHECK(PowInt(2.0f, 5) == 32.0f);
		STATIC_CHECK(PowInt(2.0, 10) == 1024.0);
		STATIC_CHECK(PowInt(2.0L, 8) == 256.0L);
	}

	SECTION("Different exponent types")
	{
		REQUIRE(PowInt(2.0, 3) == Approx(8.0));
		REQUIRE(PowInt(2.0, 3LL) == Approx(8.0));
		REQUIRE(PowInt(2.0, 3u) == Approx(8.0));
		REQUIRE(PowInt(2.0, std::int8_t{3}) == Approx(8.0));
	}

	SECTION("Agreement with std::pow")
	{
		for (int i = -10; i <= 10; i++)
		{
			REQUIRE(PowInt(2.0, i) == Approx(std::pow(2.0, i)));
			REQUIRE(PowInt(1.5, i) == Approx(std::pow(1.5, i)));
			REQUIRE(PowInt(3.0, i) == Approx(std::pow(3.0, i)));
		}
	}

	SECTION("Edge cases")
	{
		// Zero base with positive exponent
		REQUIRE(PowInt(0.0, 5) == Approx(0.0));

		// Zero base with negative exponent (should be infinity)
		REQUIRE(std::isinf(PowInt(0.0, -1)));

		// One as base
		REQUIRE(PowInt(1.0, 100) == Approx(1.0));
		REQUIRE(PowInt(1.0, -50) == Approx(1.0));

		// Negative one with even/odd exponents
		REQUIRE(PowInt(-1.0, 4) == Approx(1.0));
		REQUIRE(PowInt(-1.0, 5) == Approx(-1.0));
	}

	SECTION("Integer bases")
	{
		STATIC_CHECK(PowInt(2, 5) == 32);
		STATIC_CHECK(PowInt(3, 3) == 27);
		STATIC_CHECK(PowInt(5, 4) == 625);
		STATIC_CHECK(PowInt(10, 3) == 1000);

		// With negative exponent (should not automatically return floating point)
		{
			constexpr auto result = PowInt(2, -2);
			STATIC_CHECK(result == 0);
		}
		{
			constexpr auto result = PowInt<double>(2, -2);
			STATIC_CHECK(Abs(result - 0.25) < 1e-9);
		}
	}

	SECTION("Performance-critical small exponents")
	{
		// These should use the fast path (exponentiation by squaring)
		STATIC_CHECK(PowInt(2.0, 2) == 4.0);
		STATIC_CHECK(PowInt(2.0, 3) == 8.0);
		STATIC_CHECK(PowInt(2.0, 4) == 16.0);
		STATIC_CHECK(PowInt(2.0, 5) == 32.0);
		STATIC_CHECK(PowInt(2.0, 6) == 64.0);
		STATIC_CHECK(PowInt(2.0, 7) == 128.0);
		STATIC_CHECK(PowInt(2.0, 8) == 256.0);
		STATIC_CHECK(PowInt(2.0, 16) == 65536.0);
		STATIC_CHECK(PowInt(2.0, 32) == 4294967296.0);
	}
}
