// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Collection/IndexAccessor.hpp>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Math/EstrinPolynomial.hpp>
#include <SecUtility/Math/HornerPolynomial.hpp>
#include <SecUtility/Math/Special/Factorial.hpp>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

using namespace SecUtility;
using namespace SecUtility::Math;


//----------------------------------------------------------------------------------------------------------------------
// Test Utilities
//----------------------------------------------------------------------------------------------------------------------

namespace
{
	// Naive polynomial evaluation: ∑ coeff[i] * x^i
	template <typename Scalar, std::size_t N, typename CoeffAccessor>
	constexpr Scalar NaivePowerPolynomial(const CoeffAccessor& coeff, const Scalar x) noexcept
	{
		Scalar result = 0;

		for (std::size_t i = 0; i < N; ++i)
		{
			result += coeff(i) * std::pow(x, i);
		}

		return result;
	}


	// Naive Taylor polynomial evaluation: ∑ coeff[i] * x^i / i!
	template <typename Scalar, std::size_t N, typename CoeffAccessor>
	constexpr Scalar NaiveTaylorPolynomial(const CoeffAccessor& coeff, const Scalar x) noexcept
	{
		Scalar result = 0;

		for (std::size_t i = 0; i < N; ++i)
		{
			result += coeff(i) * std::pow(x, i) * ReciprocalFactorial(static_cast<int>(i));
		}

		return result;
	}


	// Stateful coefficient accessor for testing
	template <typename T, std::size_t N>
	struct StatefulAccessor
	{
		std::array<T, N> data{};
		mutable std::size_t access_count = 0;

		T operator()(std::size_t i) const noexcept
		{
			++access_count;
			return data[i];
		}
	};
}


//----------------------------------------------------------------------------------------------------------------------
// Basic Power Polynomial Tests
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("HornerPowerPolynomial - N=1 (constant)")
{
	constexpr std::size_t N = 1;
	constexpr double coeff[] = {5.0};

	SECTION("x = 0")
	{
		constexpr double x = 0.0;
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(5.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(5.0, 1e-10));
	}

	SECTION("x = 1")
	{
		constexpr double x = 1.0;
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(5.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(5.0, 1e-10));
	}

	SECTION("x = 2")
	{
		constexpr double x = 2.0;
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(5.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(5.0, 1e-10));
	}
}


TEST_CASE("HornerPowerPolynomial - N=2 (linear)")
{
	constexpr std::size_t N = 2;
	constexpr double coeff[] = {3.0, 2.0};  // 3 + 2x

	SECTION("x = 0")
	{
		constexpr double x = 0.0;
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(3.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(3.0, 1e-10));
	}

	SECTION("x = 1")
	{
		constexpr double x = 1.0;
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(5.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(5.0, 1e-10));
	}

	SECTION("x = 2")
	{
		constexpr double x = 2.0;
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(7.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(7.0, 1e-10));
	}

	SECTION("x = -1")
	{
		constexpr double x = -1.0;
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(1.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(1.0, 1e-10));
	}
}


TEST_CASE("HornerPowerPolynomial - N=3 (quadratic)")
{
	constexpr std::size_t N = 3;
	constexpr auto coeff = [](const auto i) { return i + 1; };  // 1 + 2x + 3x²

	SECTION("x = 0")
	{
		constexpr double x = 0.0;
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerPowerPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(1.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(1.0, 1e-10));
	}

	SECTION("x = 1")
	{
		constexpr double x = 1.0;
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerPowerPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(6.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(6.0, 1e-10));
	}

	SECTION("x = 2")
	{
		constexpr double x = 2.0;
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerPowerPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(17.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(17.0, 1e-10));
	}
}


TEST_CASE("HornerPowerPolynomial - N=5 (higher degree)")
{
	constexpr std::size_t N = 5;
	constexpr double coeff[] = {1.0, -2.0, 3.0, -4.0, 5.0};  // 1 - 2x + 3x² - 4x³ + 5x⁴

	SECTION("x = 1")
	{
		constexpr double x = 1.0;
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(3.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(3.0, 1e-10));
	}

	SECTION("x = 0.5")
	{
		constexpr double x = 0.5;
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
	}

	SECTION("x = 2")
	{
		constexpr double x = 2.0;
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Basic Taylor Polynomial Tests
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("HornerTaylorPolynomial - N=1 (constant)")
{
	constexpr std::size_t N = 1;
	constexpr double coeff[] = {5.0};

	SECTION("x = 0")
	{
		constexpr double x = 0.0;
		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaiveTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(5.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(5.0, 1e-10));
	}

	SECTION("x = 1")
	{
		constexpr double x = 1.0;
		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaiveTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(5.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(5.0, 1e-10));
	}
}


TEST_CASE("HornerTaylorPolynomial - N=2")
{
	constexpr std::size_t N = 2;
	constexpr double coeff[] = {3.0, 2.0};  // 3 + 2x/1!

	SECTION("x = 1")
	{
		constexpr double x = 1.0;
		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaiveTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(5.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(5.0, 1e-10));
	}

	SECTION("x = 2")
	{
		constexpr double x = 2.0;
		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaiveTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(7.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(7.0, 1e-10));
	}
}


TEST_CASE("HornerTaylorPolynomial - N=3")
{
	constexpr std::size_t N = 3;
	constexpr double coeff[] = {1.0, 2.0, 3.0};  // 1 + 2x/1! + 3x²/2!

	SECTION("x = 1")
	{
		constexpr double x = 1.0;
		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaiveTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(4.5, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(4.5, 1e-10));
	}

	SECTION("x = 2")
	{
		constexpr double x = 2.0;
		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaiveTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Mathematical Identities - Power Polynomial
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("HornerPowerPolynomial - Geometric series: 1 + x + x^2 + ... = 1/(1-x)")
{
	// All coefficients are 1
	constexpr auto coeff = [](std::size_t) constexpr noexcept { return 1.0; };

	SECTION("x = 0.1")
	{
		constexpr double x = 0.1;
		constexpr std::size_t N = 10;

		constexpr auto result_ct = HornerPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerPowerPolynomial<double, N>(coeff, x);  // NOLINT
		constexpr auto expected = 1.0 / (1.0 - x);  // Geometric series formula

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 0.01));  // 1% tolerance for N=10
		CHECK(result_ct > 0.0);                                             // Should be positive
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 0.01));
		CHECK(result_rt > 0.0);
	}

	SECTION("x = 0.5")
	{
		constexpr double x = 0.5;
		constexpr std::size_t N = 20;

		constexpr auto result_ct = HornerPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerPowerPolynomial<double, N>(coeff, x);  // NOLINT
		constexpr auto expected = 1.0 / (1.0 - x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 0.001));  // Better convergence with N=20
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 0.001));
	}

	SECTION("x = -0.5")
	{
		constexpr double x = -0.5;
		constexpr std::size_t N = 15;

		constexpr auto result_ct = HornerPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerPowerPolynomial<double, N>(coeff, x);  // NOLINT
		constexpr auto expected = 1.0 / (1.0 - x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 0.001));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 0.001));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Mathematical Identities - Taylor Polynomial (e^x)
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("HornerTaylorPolynomial - e^x Taylor series: all coefficients = 1")
{
	// e^x = 1 + x + x²/2! + x³/3! + ... = ∑ x^k/k!
	// So all coefficients are 1
	constexpr auto coeff = [](std::size_t) constexpr noexcept { return 1.0; };

	SECTION("x = 0.1")
	{
		constexpr double x = 0.1;
		constexpr std::size_t N = 10;

		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::exp(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
	}

	SECTION("x = 1.0")
	{
		constexpr double x = 1.0;
		constexpr std::size_t N = 15;

		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::exp(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-7));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-7));
	}

	SECTION("x = 2.0")
	{
		constexpr double x = 2.0;
		constexpr std::size_t N = 20;

		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::exp(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-6));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-6));
	}

	SECTION("x = -1.0")
	{
		constexpr double x = -1.0;
		constexpr std::size_t N = 15;

		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::exp(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-8));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-8));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Mathematical Identities - Taylor Polynomial (sin(x))
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("HornerTaylorPolynomial - sin(x) Taylor series")
{
	// sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
	// Coefficients for k=0,1,2,...: c_0=0, c_1=1, c_2=0, c_3=-1, c_4=0, c_5=1, ...
	constexpr auto coeff = [](const std::size_t i) constexpr noexcept
	{
		if (i % 2 == 0)
		{
			return 0;
		}

		return (i / 2) % 2 == 0 ? 1 : -1;
	};

	SECTION("x = 0.1")
	{
		constexpr double x = 0.1;
		constexpr std::size_t N = 8;

		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::sin(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
	}

	SECTION("x = 1.0")
	{
		constexpr double x = 1.0;
		constexpr std::size_t N = 8;

		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::sin(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-5));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-5));
	}

	SECTION("x = π/6 (30 degrees)")
	{
		constexpr double x = SecUtility::Math::Constant::Pi<double> / 6.0;
		constexpr std::size_t N = 8;

		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::sin(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-5));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-5));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Mathematical Identities - Taylor Polynomial (cos(x))
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("HornerTaylorPolynomial - cos(x) Taylor series")
{
	// cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
	// Coefficients for k=0,1,2,...: c_0=1, c_1=0, c_2=-1, c_3=0, c_4=1, c_5=0, c_6=-1, ...
	constexpr auto coeff = [](const std::size_t i) constexpr noexcept
	{
		if (i % 2 == 0)
		{
			return (i / 2) % 2 == 0 ? 1 : -1;
		}
		return 0;
	};

	SECTION("x = 0.1")
	{
		constexpr double x = 0.1;
		constexpr std::size_t N = 9;

		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::cos(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
	}

	SECTION("x = 1.0")
	{
		constexpr double x = 1.0;
		constexpr std::size_t N = 9;

		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::cos(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-5));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-5));
	}

	SECTION("x = π/3 (60 degrees)")
	{
		constexpr double x = SecUtility::Math::Constant::Pi<double> / 3.0;
		constexpr std::size_t N = 10;

		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::cos(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-4));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-4));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Implementation Consistency Tests
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("HornerPowerPolynomial - Unrolled vs Loop consistency")
{
	constexpr std::size_t SmallN = 10;
	constexpr std::size_t BoundaryN = 32;
	constexpr std::size_t LargeN = 50;

	constexpr auto coeff = [](const auto i) { return i + 1; };

	SECTION("N = 10 (unrolled)")
	{
		constexpr double x = 1.5;
		const auto result_unrolled = UnrolledHornerPowerPolynomial<double, SmallN>(coeff, x);
		const auto result_loop =
		        Math::Detail::HornerPolynomial::HornerPowerPolynomial_Loop<double, SmallN>(coeff, x);

		CHECK_THAT(result_unrolled, Catch::Matchers::WithinRel(result_loop, 1e-15));
	}

	SECTION("N = 32 (unrolled boundary)")
	{
		constexpr double x = 1.5;
		const auto result_unrolled = UnrolledHornerPowerPolynomial<double, BoundaryN>(coeff, x);
		const auto result_loop =
		        Math::Detail::HornerPolynomial::HornerPowerPolynomial_Loop<double, BoundaryN>(coeff, x);

		CHECK_THAT(result_unrolled, Catch::Matchers::WithinRel(result_loop, 1e-15));
	}

	SECTION("N = 50 (loop)")
	{
		constexpr double x = 1.5;
		const auto result_loop =
		        Math::Detail::HornerPolynomial::HornerPowerPolynomial_Loop<double, LargeN>(coeff, x);
		const auto result_public = HornerPowerPolynomial<double, LargeN>(coeff, x);

		CHECK_THAT(result_loop, Catch::Matchers::WithinRel(result_public, 1e-15));
	}
}


TEST_CASE("HornerTaylorPolynomial - Unrolled vs Loop consistency")
{
	constexpr std::size_t SmallN = 10;
	constexpr std::size_t BoundaryN = 32;
	constexpr std::size_t LargeN = 50;

	constexpr auto coeff = [](const auto i) { return i + 1; };

	SECTION("N = 10 (unrolled)")
	{
		constexpr double x = 1.5;
		const auto result_unrolled = UnrolledHornerTaylorPolynomial<double, SmallN>(coeff, x);
		const auto result_loop =
		        Math::Detail::HornerPolynomial::HornerTaylorPolynomial_Loop<double, SmallN>(coeff, x);

		CHECK_THAT(result_unrolled, Catch::Matchers::WithinRel(result_loop, 1e-15));
	}

	SECTION("N = 32 (unrolled boundary)")
	{
		constexpr double x = 1.5;
		const auto result_unrolled = UnrolledHornerTaylorPolynomial<double, BoundaryN>(coeff, x);
		const auto result_loop =
		        Math::Detail::HornerPolynomial::HornerTaylorPolynomial_Loop<double, BoundaryN>(coeff, x);

		CHECK_THAT(result_unrolled, Catch::Matchers::WithinRel(result_loop, 1e-15));
	}

	SECTION("N = 50 (loop)")
	{
		constexpr double x = 1.5;
		const auto result_loop =
		        Math::Detail::HornerPolynomial::HornerTaylorPolynomial_Loop<double, LargeN>(coeff, x);
		const auto result_public = HornerTaylorPolynomial<double, LargeN>(coeff, x);

		CHECK_THAT(result_loop, Catch::Matchers::WithinRel(result_public, 1e-15));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Boundary and Edge Cases
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("HornerPowerPolynomial - Edge cases")
{
	SECTION("delta = 0")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto i) { return i + 1; };
		constexpr double x = 0.0;

		constexpr auto result_ct = HornerPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerPowerPolynomial<double, N>(coeff, x);  // NOLINT

		// Should return just the constant term
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(1.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(1.0, 1e-10));
	}

	SECTION("delta = 1")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto i) { return i + 1; };
		constexpr double x = 1.0;

		constexpr auto result_ct = HornerPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerPowerPolynomial<double, N>(coeff, x);  // NOLINT

		// Should return sum of all coefficients
		constexpr auto expected = 1.0 + 2.0 + 3.0 + 4.0 + 5.0;
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(expected, 1e-10));
	}

	SECTION("All zero coefficients")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto) { return 0.0; };
		constexpr double x = 2.5;

		constexpr auto result_ct = HornerPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerPowerPolynomial<double, N>(coeff, x);  // NOLINT

		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(0.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(0.0, 1e-10));
	}

	SECTION("All one coefficients")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto) { return 1.0; };
		constexpr double x = 2.0;

		constexpr auto result_ct = HornerPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerPowerPolynomial<double, N>(coeff, x);  // NOLINT

		// 1 + 2 + 4 + 8 + 16 = 31
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(31.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(31.0, 1e-10));
	}
}


TEST_CASE("HornerTaylorPolynomial - Edge cases")
{
	SECTION("delta = 0")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto i) { return i + 1; };
		constexpr double x = 0.0;

		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT

		// Should return just the constant term
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(1.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(1.0, 1e-10));
	}

	SECTION("All zero coefficients")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto) { return 0.0; };
		constexpr double x = 2.5;

		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT

		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(0.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(0.0, 1e-10));
	}

	SECTION("All one coefficients (e^x approximation)")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto) { return 1.0; };
		constexpr double x = 1.0;

		constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT

		// 1 + 1 + 1/2 + 1/6 + 1/24 = 2.70833...
		constexpr double expected = 1.0 + 1.0 + 1.0 / 2.0 + 1.0 / 6.0 + 1.0 / 24.0;
		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Different Coefficient Accessor Types
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("HornerPowerPolynomial - Different accessor types")
{
	SECTION("C-array")
	{
		constexpr std::size_t N = 3;
		constexpr double coeff[] = {1.0, -2.0, 3.0};  // Non-sequential values
		constexpr double x = 1.5;

		constexpr auto result_ct = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = HornerPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
	}

	SECTION("Lambda (lazy coefficient generation)")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff_lambda = [](const std::size_t i) constexpr noexcept { return static_cast<double>(i + 1); };
		constexpr double x = 1.5;

		constexpr auto result_ct = HornerPowerPolynomial<double, N>(coeff_lambda, x);
		const auto result_rt = HornerPowerPolynomial<double, N>(coeff_lambda, x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(coeff_lambda, x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
	}

	SECTION("Stateful functor")
	{
		constexpr std::size_t N = 5;
		StatefulAccessor<double, N> functor{};
		functor.data = {1.0, 2.0, 3.0, 4.0, 5.0};
		constexpr double x = 1.5;

		const auto result = HornerPowerPolynomial<double, N>(functor, x);
		CHECK(functor.access_count == N);  // Functor was actually called

		const auto expected = NaivePowerPolynomial<double, N>(functor, x);
		CHECK_THAT(result, Catch::Matchers::WithinRel(expected, 1e-15));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Different Scalar Types
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("HornerPowerPolynomial - Different scalar types")
{
	constexpr std::size_t N = 3;
	constexpr double x = 1.5;

	SECTION("float")
	{
		constexpr auto coeff = [](const auto i) { return static_cast<float>(i + 1); };
		constexpr auto result_ct = HornerPowerPolynomial<float, N>(coeff, x);
		const auto result_rt = HornerPowerPolynomial<float, N>(coeff, x);  // NOLINT
		const auto expected = NaivePowerPolynomial<float, N>(coeff, x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-6f));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-6f));
	}

	SECTION("double")
	{
		constexpr auto coeff = [](const auto i) { return i + 1; };
		constexpr auto result_ct = HornerPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = HornerPowerPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
	}

	SECTION("long double")
	{
		constexpr auto coeff = [](const auto i) { return static_cast<long double>(i + 1); };
		constexpr auto result_ct = HornerPowerPolynomial<long double, N>(coeff, x);
		const auto result_rt = HornerPowerPolynomial<long double, N>(coeff, x);  // NOLINT
		const auto expected = NaivePowerPolynomial<long double, N>(coeff, x);

		// STATIC_CHECK(Abs(result - expected) < 1e-18);
		CHECK(Abs(result_ct - expected) < 1e-18);
		CHECK(Abs(result_rt - expected) < 1e-18);
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Constexpr Tests
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("HornerPowerPolynomial - Constexpr evaluation")
{
	constexpr std::size_t N = 3;
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 2.0;

	constexpr auto result_ct = HornerPowerPolynomial<double, N>(coeff, x);
	const auto result_rt = HornerPowerPolynomial<double, N>(coeff, x);  // NOLINT
	constexpr double expected = 17.0;  // 1 + 2*2 + 3*4 = 17

	// STATIC_CHECK(Abs(result - expected) < 1e-18);
	CHECK(Abs(result_ct - expected) < 1e-18);
	CHECK(Abs(result_rt - expected) < 1e-18);
}


TEST_CASE("HornerTaylorPolynomial - Constexpr evaluation")
{
	constexpr std::size_t N = 3;
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 2;

	constexpr auto result_ct = HornerTaylorPolynomial<double, N>(coeff, x);
	const auto result_rt = HornerTaylorPolynomial<double, N>(coeff, x);  // NOLINT
	constexpr double expected = 11;

	// STATIC_CHECK(Abs(result - expected) < 1e-18);
	CHECK(Abs(result_ct - expected) < 1e-18);
	CHECK(Abs(result_rt - expected) < 1e-18);
}


//----------------------------------------------------------------------------------------------------------------------
// Detail Function Tests (Internal Implementation)
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("Detail::HornerPolynomial - HornerPowerPolynomial_Loop")
{
	constexpr std::size_t N = 10;
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 1.5;

	constexpr auto result_ct = Math::Detail::HornerPolynomial::HornerPowerPolynomial_Loop<double, N>(coeff, x);
	const auto result_rt = Math::Detail::HornerPolynomial::HornerPowerPolynomial_Loop<double, N>(coeff, x);  // NOLINT
	const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

	CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
	CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
}


TEST_CASE("Detail::HornerPolynomial - HornePowerPolynomial_Fold")
{
	constexpr std::size_t N = 10;
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 1.5;

	constexpr auto result_ct = Math::Detail::HornerPolynomial::HornerPowerPolynomial_Fold<double, N>(
	        coeff, x, MakeReversedIndexSequence<N>{});
	const auto result_rt = Math::Detail::HornerPolynomial::HornerPowerPolynomial_Fold<double, N>(  // NOLINT
	        coeff, x, MakeReversedIndexSequence<N>{});
	const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

	CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
	CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
}


TEST_CASE("Detail::HornerPolynomial - HornerTaylorPolynomial_Loop")
{
	constexpr std::size_t N = 10;
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 1.5;

	constexpr auto result_ct = Math::Detail::HornerPolynomial::HornerTaylorPolynomial_Loop<double, N>(coeff, x);
	const auto result_rt = Math::Detail::HornerPolynomial::HornerTaylorPolynomial_Loop<double, N>(coeff, x);  // NOLINT
	const auto expected = NaiveTaylorPolynomial<double, N>(coeff, x);

	CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
	CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
}


TEST_CASE("Detail::HornerPolynomial - HornerTaylorPolynomial_Fold")
{
	constexpr std::size_t N = 10;
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 1.5;

	constexpr auto result_ct = Math::Detail::HornerPolynomial::HornerTaylorPolynomial_Fold<double, N>(
	        coeff, x, MakeReversedIndexSequence<N>{});
	const auto result_rt = Math::Detail::HornerPolynomial::HornerTaylorPolynomial_Fold<double, N>(  // NOLINT
	        coeff, x, MakeReversedIndexSequence<N>{});
	const auto expected = NaiveTaylorPolynomial<double, N>(coeff, x);

	CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
	CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
}


//======================================================================================================================
// Estrin Polynomial Tests
//======================================================================================================================

//----------------------------------------------------------------------------------------------------------------------
// Estrin Power Polynomial - Basic Tests
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("EstrinPowerPolynomial - N=1 (constant)")
{
	constexpr std::size_t N = 1;
	constexpr double coeff[] = {5.0};

	SECTION("x = 0")
	{
		constexpr double x = 0.0;
		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(5.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(5.0, 1e-10));
	}

	SECTION("x = 2")
	{
		constexpr double x = 2.0;
		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT

		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(5.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(5.0, 1e-10));
	}
}


TEST_CASE("EstrinPowerPolynomial - N=2 (linear)")
{
	constexpr std::size_t N = 2;
	constexpr double coeff[] = {3.0, 2.0};  // 3 + 2x

	SECTION("x = 2")
	{
		constexpr double x = 2.0;
		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT

		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(7.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(7.0, 1e-10));
	}

	SECTION("x = -1")
	{
		constexpr double x = -1.0;
		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT

		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(1.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(1.0, 1e-10));
	}
}


TEST_CASE("EstrinPowerPolynomial - N=3 (quadratic, odd N)")
{
	constexpr std::size_t N = 3;
	constexpr auto coeff = [](const auto i) { return i + 1; };  // 1 + 2x + 3x²

	SECTION("x = 2")
	{
		constexpr double x = 2.0;
		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(coeff, x);  // NOLINT

		// 1 + 2·2 + 3·4 = 17
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(17.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(17.0, 1e-10));
	}

	SECTION("x = 0.5")
	{
		constexpr double x = 0.5;
		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
	}
}


TEST_CASE("EstrinPowerPolynomial - N=4 (cubic, even N)")
{
	constexpr std::size_t N = 4;
	constexpr auto coeff = [](const auto i) { return i + 1; };  // 1 + 2x + 3x² + 4x³

	SECTION("x = 2")
	{
		constexpr double x = 2.0;
		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(coeff, x);  // NOLINT

		// 1 + 4 + 12 + 32 = 49
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(49.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(49.0, 1e-10));
	}
}


TEST_CASE("EstrinPowerPolynomial - N=5 (quartic, odd N)")
{
	constexpr std::size_t N = 5;
	constexpr double coeff[] = {1.0, -2.0, 3.0, -4.0, 5.0};  // 1 - 2x + 3x² - 4x³ + 5x⁴

	SECTION("x = 1")
	{
		constexpr double x = 1.0;
		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
	}

	SECTION("x = 2")
	{
		constexpr double x = 2.0;
		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
	}
}


TEST_CASE("EstrinPowerPolynomial - N=7 (odd, multi-level recursion)")
{
	// 7 → 4 → 2: exercises odd N at the top level only.
	constexpr std::size_t N = 7;
	constexpr auto coeff = [](const auto i) { return i + 1; };

	SECTION("x = 1.5")
	{
		constexpr double x = 1.5;
		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
	}
}


TEST_CASE("EstrinPowerPolynomial - N=11 (odd at multiple recursion levels)")
{
	// 11 → 6 → 3 → 2: odd N appears at levels 0 and 2, with even N=6 in between.
	// This stresses the if-constexpr odd-N branch in the recursive helper.
	constexpr std::size_t N = 11;
	constexpr auto coeff = [](const auto i) { return i + 1; };

	SECTION("x = 1.3")
	{
		constexpr double x = 1.3;
		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-14));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-14));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Estrin Taylor Polynomial - Basic Tests
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("EstrinTaylorPolynomial - N=1 (constant)")
{
	constexpr std::size_t N = 1;
	constexpr double coeff[] = {5.0};

	SECTION("x = 0")
	{
		constexpr double x = 0.0;
		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT

		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(5.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(5.0, 1e-10));
	}
}


TEST_CASE("EstrinTaylorPolynomial - N=2")
{
	constexpr std::size_t N = 2;
	constexpr double coeff[] = {3.0, 2.0};  // 3 + 2x/1!

	SECTION("x = 2")
	{
		constexpr double x = 2.0;
		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT

		// 3 + 2·2/1! = 7
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(7.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(7.0, 1e-10));
	}
}


TEST_CASE("EstrinTaylorPolynomial - N=3")
{
	constexpr std::size_t N = 3;
	constexpr double coeff[] = {1.0, 2.0, 3.0};  // 1 + 2x/1! + 3x²/2!

	SECTION("x = 1")
	{
		constexpr double x = 1.0;
		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT

		// 1 + 2 + 3/2 = 4.5
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(4.5, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(4.5, 1e-10));
	}

	SECTION("x = 2")
	{
		constexpr double x = 2.0;
		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaiveTaylorPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Estrin Mathematical Identities - Power Polynomial (Geometric series)
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("EstrinPowerPolynomial - Geometric series: 1 + x + x^2 + ... = 1/(1-x)")
{
	constexpr auto coeff = [](std::size_t) constexpr noexcept { return 1.0; };

	SECTION("x = 0.5, N = 20")
	{
		constexpr double x = 0.5;
		constexpr std::size_t N = 20;

		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(coeff, x);  // NOLINT
		constexpr auto expected = 1.0 / (1.0 - x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 0.001));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 0.001));
	}

	SECTION("x = -0.5, N = 15")
	{
		constexpr double x = -0.5;
		constexpr std::size_t N = 15;

		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(coeff, x);  // NOLINT
		constexpr auto expected = 1.0 / (1.0 - x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 0.001));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 0.001));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Estrin Mathematical Identities - Taylor Polynomial (e^x)
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("EstrinTaylorPolynomial - e^x Taylor series: all coefficients = 1")
{
	constexpr auto coeff = [](std::size_t) constexpr noexcept { return 1.0; };

	SECTION("x = 0.1, N = 10")
	{
		constexpr double x = 0.1;
		constexpr std::size_t N = 10;

		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::exp(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
	}

	SECTION("x = 1.0, N = 15")
	{
		constexpr double x = 1.0;
		constexpr std::size_t N = 15;

		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::exp(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-7));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-7));
	}

	SECTION("x = -1.0, N = 15")
	{
		constexpr double x = -1.0;
		constexpr std::size_t N = 15;

		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::exp(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-8));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-8));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Estrin Mathematical Identities - Taylor Polynomial (sin(x))
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("EstrinTaylorPolynomial - sin(x) Taylor series")
{
	// sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
	constexpr auto coeff = [](const std::size_t i) constexpr noexcept
	{
		if (i % 2 == 0)
		{
			return 0.0;
		}

		return (i / 2) % 2 == 0 ? 1.0 : -1.0;
	};

	SECTION("x = 0.1, N = 8")
	{
		constexpr double x = 0.1;
		constexpr std::size_t N = 8;

		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::sin(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
	}

	SECTION("x = 1.0, N = 8")
	{
		constexpr double x = 1.0;
		constexpr std::size_t N = 8;

		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::sin(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-5));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-5));
	}

	SECTION("x = π/6 (30 degrees), N = 8")
	{
		constexpr double x = SecUtility::Math::Constant::Pi<double> / 6.0;
		constexpr std::size_t N = 8;

		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::sin(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-5));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-5));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Estrin Mathematical Identities - Taylor Polynomial (cos(x))
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("EstrinTaylorPolynomial - cos(x) Taylor series")
{
	// cos(x) = 1 - x²/2! + x⁴/4! - x⁶/6! + ...
	constexpr auto coeff = [](const std::size_t i) constexpr noexcept
	{
		if (i % 2 == 0)
		{
			return (i / 2) % 2 == 0 ? 1.0 : -1.0;
		}
		return 0.0;
	};

	SECTION("x = 0.1, N = 9")
	{
		constexpr double x = 0.1;
		constexpr std::size_t N = 9;

		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::cos(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
	}

	SECTION("x = 1.0, N = 9")
	{
		constexpr double x = 1.0;
		constexpr std::size_t N = 9;

		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = std::cos(x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-5));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-5));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Estrin Implementation Consistency - Unrolled vs Loop
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("EstrinPowerPolynomial - Unrolled vs Loop consistency")
{
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 1.5;

	SECTION("N = 5 (odd, unrolled)")
	{
		constexpr std::size_t N = 5;
		const auto result_unrolled = UnrolledEstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_loop =
		        Math::Detail::EstrinPolynomial::EstrinPowerPolynomial_Loop<double, N>(coeff, x);

		CHECK_THAT(result_unrolled, Catch::Matchers::WithinRel(result_loop, 1e-15));
	}

	SECTION("N = 16 (even power of 2, unrolled)")
	{
		constexpr std::size_t N = 16;
		const auto result_unrolled = UnrolledEstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_loop =
		        Math::Detail::EstrinPolynomial::EstrinPowerPolynomial_Loop<double, N>(coeff, x);

		CHECK_THAT(result_unrolled, Catch::Matchers::WithinRel(result_loop, 1e-15));
	}

	SECTION("N = 17 (odd, unrolled)")
	{
		constexpr std::size_t N = 17;
		const auto result_unrolled = UnrolledEstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_loop =
		        Math::Detail::EstrinPolynomial::EstrinPowerPolynomial_Loop<double, N>(coeff, x);

		CHECK_THAT(result_unrolled, Catch::Matchers::WithinRel(result_loop, 1e-15));
	}

	SECTION("N = 32 (unrolled boundary)")
	{
		constexpr std::size_t N = 32;
		const auto result_unrolled = UnrolledEstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_loop =
		        Math::Detail::EstrinPolynomial::EstrinPowerPolynomial_Loop<double, N>(coeff, x);

		CHECK_THAT(result_unrolled, Catch::Matchers::WithinRel(result_loop, 1e-15));
	}

	SECTION("N = 33 (loop vs naive)")
	{
		// Over the unroll boundary: EstrinPowerPolynomial dispatches to Loop.
		// Compare against naive since UnrolledEstrinPowerPolynomial<N> is not
		// instantiated for N > 32, so loop-vs-public would be tautological.
		constexpr std::size_t N = 33;
		const auto result = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

		CHECK_THAT(result, Catch::Matchers::WithinRel(expected, 1e-12));
	}

	SECTION("N = 50 (loop vs naive)")
	{
		constexpr std::size_t N = 50;
		const auto result = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

		CHECK_THAT(result, Catch::Matchers::WithinRel(expected, 1e-12));
	}
}


TEST_CASE("EstrinTaylorPolynomial - Unrolled vs Loop consistency")
{
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 1.5;

	SECTION("N = 17 (odd, unrolled)")
	{
		constexpr std::size_t N = 17;
		const auto result_unrolled = UnrolledEstrinTaylorPolynomial<double, N>(coeff, x);
		const auto result_loop =
		        Math::Detail::EstrinPolynomial::EstrinTaylorPolynomial_Loop<double, N>(coeff, x);

		CHECK_THAT(result_unrolled, Catch::Matchers::WithinRel(result_loop, 1e-15));
	}

	SECTION("N = 32 (unrolled boundary)")
	{
		constexpr std::size_t N = 32;
		const auto result_unrolled = UnrolledEstrinTaylorPolynomial<double, N>(coeff, x);
		const auto result_loop =
		        Math::Detail::EstrinPolynomial::EstrinTaylorPolynomial_Loop<double, N>(coeff, x);

		CHECK_THAT(result_unrolled, Catch::Matchers::WithinRel(result_loop, 1e-15));
	}

	SECTION("N = 50 (loop vs naive)")
	{
		// Over the unroll boundary: compare against naive, since loop-vs-public
		// would be tautological for N > 32.
		constexpr std::size_t N = 50;
		const auto result = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto expected = NaiveTaylorPolynomial<double, N>(coeff, x);

		CHECK_THAT(result, Catch::Matchers::WithinRel(expected, 1e-12));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Estrin vs Horner Cross-Algorithm Consistency
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("Estrin vs Horner - Power Polynomial cross-algorithm")
{
	// Both schemes evaluate the same polynomial; only FP rounding differs.
	// For well-conditioned cases agreement is to a few ULP.
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 1.5;

	SECTION("N = 5 (odd)")
	{
		constexpr std::size_t N = 5;
		const auto estrin = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto horner = HornerPowerPolynomial<double, N>(coeff, x);

		CHECK_THAT(estrin, Catch::Matchers::WithinRel(horner, 1e-14));
	}

	SECTION("N = 11 (odd at multiple recursion levels)")
	{
		constexpr std::size_t N = 11;
		const auto estrin = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto horner = HornerPowerPolynomial<double, N>(coeff, x);

		CHECK_THAT(estrin, Catch::Matchers::WithinRel(horner, 1e-13));
	}

	SECTION("N = 32 (unrolled boundary)")
	{
		constexpr std::size_t N = 32;
		const auto estrin = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto horner = HornerPowerPolynomial<double, N>(coeff, x);

		CHECK_THAT(estrin, Catch::Matchers::WithinRel(horner, 1e-13));
	}
}


TEST_CASE("Estrin vs Horner - Taylor Polynomial cross-algorithm")
{
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 1.5;

	SECTION("N = 11")
	{
		constexpr std::size_t N = 11;
		const auto estrin = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto horner = HornerTaylorPolynomial<double, N>(coeff, x);

		CHECK_THAT(estrin, Catch::Matchers::WithinRel(horner, 1e-13));
	}

	SECTION("N = 32")
	{
		constexpr std::size_t N = 32;
		const auto estrin = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto horner = HornerTaylorPolynomial<double, N>(coeff, x);

		CHECK_THAT(estrin, Catch::Matchers::WithinRel(horner, 1e-13));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Estrin Boundary and Edge Cases
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("EstrinPowerPolynomial - Edge cases")
{
	SECTION("delta = 0")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto i) { return i + 1; };
		constexpr double x = 0.0;

		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(coeff, x);  // NOLINT

		// Constant term only
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(1.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(1.0, 1e-10));
	}

	SECTION("delta = 1")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto i) { return i + 1; };
		constexpr double x = 1.0;

		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(coeff, x);  // NOLINT

		// Sum of coefficients: 1+2+3+4+5 = 15
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(15.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(15.0, 1e-10));
	}

	SECTION("All zero coefficients")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto) { return 0.0; };
		constexpr double x = 2.5;

		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(coeff, x);  // NOLINT

		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(0.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(0.0, 1e-10));
	}

	SECTION("All one coefficients, N = 5")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto) { return 1.0; };
		constexpr double x = 2.0;

		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(coeff, x);  // NOLINT

		// 1 + 2 + 4 + 8 + 16 = 31
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(31.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(31.0, 1e-10));
	}
}


TEST_CASE("EstrinTaylorPolynomial - Edge cases")
{
	SECTION("delta = 0")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto i) { return i + 1; };
		constexpr double x = 0.0;

		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(coeff, x);  // NOLINT

		// Constant term only
		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(1.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(1.0, 1e-10));
	}

	SECTION("All zero coefficients")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto) { return 0.0; };
		constexpr double x = 2.5;

		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(coeff, x);  // NOLINT

		CHECK_THAT(result_ct, Catch::Matchers::WithinAbs(0.0, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinAbs(0.0, 1e-10));
	}

	SECTION("All one coefficients (e^x approximation)")
	{
		constexpr std::size_t N = 5;
		constexpr auto coeff = [](const auto) { return 1.0; };
		constexpr double x = 1.0;

		constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinTaylorPolynomial<double, N>(coeff, x);  // NOLINT

		// 1 + 1 + 1/2 + 1/6 + 1/24 = 2.70833...
		constexpr double expected = 1.0 + 1.0 + 1.0 / 2.0 + 1.0 / 6.0 + 1.0 / 24.0;
		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-10));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-10));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Estrin Different Coefficient Accessor Types
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("EstrinPowerPolynomial - Different accessor types")
{
	SECTION("C-array")
	{
		constexpr std::size_t N = 5;
		constexpr double coeff[] = {1.0, -2.0, 3.0, -4.0, 5.0};
		constexpr double x = 1.5;

		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(MakeIndexAccessor(coeff), x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
	}

	SECTION("Lambda (lazy coefficient generation)")
	{
		constexpr std::size_t N = 7;
		constexpr auto coeff_lambda = [](const std::size_t i) constexpr noexcept
		{ return static_cast<double>(i + 1); };
		constexpr double x = 1.5;

		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff_lambda, x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(coeff_lambda, x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(coeff_lambda, x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
	}

	SECTION("Stateful functor")
	{
		constexpr std::size_t N = 5;
		StatefulAccessor<double, N> functor{};
		functor.data = {1.0, 2.0, 3.0, 4.0, 5.0};
		constexpr double x = 1.5;

		const auto result = EstrinPowerPolynomial<double, N>(functor, x);
		// Estrin accesses each coefficient exactly once, just like Horner;
		// the tree only re-pairs accessor outputs, not the underlying calls.
		CHECK(functor.access_count == N);

		const auto expected = NaivePowerPolynomial<double, N>(functor, x);
		CHECK_THAT(result, Catch::Matchers::WithinRel(expected, 1e-15));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Estrin Different Scalar Types
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("EstrinPowerPolynomial - Different scalar types")
{
	constexpr std::size_t N = 5;
	constexpr double x = 1.5;

	SECTION("float")
	{
		constexpr auto coeff = [](const auto i) { return static_cast<float>(i + 1); };
		constexpr auto result_ct = EstrinPowerPolynomial<float, N>(coeff, static_cast<float>(x));
		const auto result_rt = EstrinPowerPolynomial<float, N>(coeff, static_cast<float>(x));  // NOLINT
		const auto expected = NaivePowerPolynomial<float, N>(coeff, static_cast<float>(x));

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-5f));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-5f));
	}

	SECTION("double")
	{
		constexpr auto coeff = [](const auto i) { return i + 1; };
		constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff, x);
		const auto result_rt = EstrinPowerPolynomial<double, N>(coeff, x);  // NOLINT
		const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

		CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
		CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
	}

	SECTION("long double")
	{
		constexpr auto coeff = [](const auto i) { return static_cast<long double>(i + 1); };
		constexpr auto result_ct =
		        EstrinPowerPolynomial<long double, N>(coeff, static_cast<long double>(x));
		const auto result_rt =  // NOLINT
		        EstrinPowerPolynomial<long double, N>(coeff, static_cast<long double>(x));
		const auto expected =
		        NaivePowerPolynomial<long double, N>(coeff, static_cast<long double>(x));

		CHECK(Abs(result_ct - expected) < 1e-18);
		CHECK(Abs(result_rt - expected) < 1e-18);
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Estrin Constexpr Tests
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("EstrinPowerPolynomial - Constexpr evaluation")
{
	constexpr std::size_t N = 5;
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 2.0;

	constexpr auto result_ct = EstrinPowerPolynomial<double, N>(coeff, x);
	const auto result_rt = EstrinPowerPolynomial<double, N>(coeff, x);  // NOLINT

	// 1 + 2·2 + 3·4 + 4·8 + 5·16 = 129
	constexpr double expected = 129.0;

	CHECK(Abs(result_ct - expected) < 1e-18);
	CHECK(Abs(result_rt - expected) < 1e-18);
}


TEST_CASE("EstrinTaylorPolynomial - Constexpr evaluation")
{
	constexpr std::size_t N = 3;
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 2.0;

	constexpr auto result_ct = EstrinTaylorPolynomial<double, N>(coeff, x);
	const auto result_rt = EstrinTaylorPolynomial<double, N>(coeff, x);  // NOLINT

	// 1 + 2·2/1! + 3·4/2! = 1 + 4 + 6 = 11
	constexpr double expected = 11.0;

	CHECK(Abs(result_ct - expected) < 1e-18);
	CHECK(Abs(result_rt - expected) < 1e-18);
}


//----------------------------------------------------------------------------------------------------------------------
// Estrin Detail Function Tests (Internal Implementation)
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("Detail::EstrinPolynomial - EstrinPowerPolynomial_Loop")
{
	constexpr std::size_t N = 10;
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 1.5;

	constexpr auto result_ct = Math::Detail::EstrinPolynomial::EstrinPowerPolynomial_Loop<double, N>(coeff, x);
	const auto result_rt = Math::Detail::EstrinPolynomial::EstrinPowerPolynomial_Loop<double, N>(coeff, x);  // NOLINT
	const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

	CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
	CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
}


TEST_CASE("Detail::EstrinPolynomial - EstrinPowerPolynomial_Recursive")
{
	constexpr std::size_t N = 10;
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 1.5;

	constexpr auto result_ct =
	        Math::Detail::EstrinPolynomial::EstrinPowerPolynomial_Recursive<double, N>(coeff, x);
	const auto result_rt =  // NOLINT
	        Math::Detail::EstrinPolynomial::EstrinPowerPolynomial_Recursive<double, N>(coeff, x);
	const auto expected = NaivePowerPolynomial<double, N>(coeff, x);

	CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
	CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
}


TEST_CASE("Detail::EstrinPolynomial - EstrinTaylorPolynomial_Loop")
{
	constexpr std::size_t N = 10;
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 1.5;

	constexpr auto result_ct =
	        Math::Detail::EstrinPolynomial::EstrinTaylorPolynomial_Loop<double, N>(coeff, x);
	const auto result_rt =  // NOLINT
	        Math::Detail::EstrinPolynomial::EstrinTaylorPolynomial_Loop<double, N>(coeff, x);
	const auto expected = NaiveTaylorPolynomial<double, N>(coeff, x);

	CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
	CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
}


TEST_CASE("Detail::EstrinPolynomial - EstrinTaylorPolynomial_Recursive")
{
	constexpr std::size_t N = 10;
	constexpr auto coeff = [](const auto i) { return i + 1; };
	constexpr double x = 1.5;

	constexpr auto result_ct =
	        Math::Detail::EstrinPolynomial::EstrinTaylorPolynomial_Recursive<double, N>(coeff, x);
	const auto result_rt =  // NOLINT
	        Math::Detail::EstrinPolynomial::EstrinTaylorPolynomial_Recursive<double, N>(coeff, x);
	const auto expected = NaiveTaylorPolynomial<double, N>(coeff, x);

	CHECK_THAT(result_ct, Catch::Matchers::WithinRel(expected, 1e-15));
	CHECK_THAT(result_rt, Catch::Matchers::WithinRel(expected, 1e-15));
}
