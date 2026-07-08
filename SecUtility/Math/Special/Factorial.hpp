// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 Andy Brown

#pragma once

#include <SecUtility/Raw/Int.hpp>
#include <SecUtility/Macro/ForceInline.hpp>
#include <type_traits>
#include <array>
#include <cassert>


namespace SecUtility::Math
{
	// Two overloads: explicit return type via CalculateFactorial<T>(n), or T deduced from n.
	// The empty pack between Scalar and Int permits explicit Scalar specification while still deducing Int from n.
	/// <summary>
	/// Computes n! for non-negative integers.
	/// </summary>
	/// <remarks>
	/// Return type is the explicit template argument Scalar when supplied, otherwise deduced from n.
	/// Requires n &gt;= 0 (asserted) and Int to be an integral type (static_asserted).
	/// </remarks>
	template <typename Scalar, typename..., typename Int>
	constexpr std::enable_if_t<!std::is_same_v<Scalar, void>, Scalar> CalculateFactorial(const Int n) noexcept
	{
		static_assert(std::is_integral_v<Int>);
		assert(n >= 0);
		Scalar result = 1;
		for (Int i = 2; i <= n; i++)
		{
			result *= static_cast<Scalar>(i);
		}
		return result;
	}

	template <typename X = void, typename..., typename Int>
	constexpr std::enable_if_t<std::is_same_v<X, void>, Int> CalculateFactorial(const Int n) noexcept
	{
		return CalculateFactorial<Int>(n);
	}

	/// <summary>
	/// Computes n!! (double factorial). Accepts non-negative integers and negative odd integers.
	/// </summary>
	/// <remarks>
	/// Return type is the explicit template argument Scalar when supplied, otherwise deduced from n.
	/// Requires Int to be an integral type (static_asserted) and n to be odd when negative (asserted).
	/// For negative odd n &lt;= -5 the true value is fractional and integer division truncates it to 0.
	/// </remarks>
	template <typename Scalar, typename..., typename Int>
	constexpr std::enable_if_t<!std::is_same_v<Scalar, void>, Scalar> CalculateDoubleFactorial(const Int n) noexcept
	{
		static_assert(std::is_integral_v<Int>);
		if (n < 0)
		{
			assert(n % 2 != 0);
			const auto sign = (n - 1) % 4 == 0 ? 1 : -1;
			return static_cast<Scalar>(sign * n) / CalculateDoubleFactorial<Scalar>(-n);
		}

		Scalar result = 1;
		for (Int i = n; i > 1; i -= 2)
		{
			result *= static_cast<Scalar>(i);
		}
		return result;
	}

	template <typename X = void, typename..., typename Int>
	constexpr std::enable_if_t<std::is_same_v<X, void>, Int> CalculateDoubleFactorial(const Int n) noexcept
	{
		return CalculateDoubleFactorial<Int>(n);
	}

	namespace Detail::Factorial
	{
		// 20! = 2,432,902,008,176,640,000 is the largest factorial that fits in Int64; 21! would overflow.
		inline constexpr auto Factorials = []() constexpr
		{
			std::array<Int64, 21> data = {};

			for (int i = 0; i < 21; i++)
			{
				data[i] = CalculateFactorial<Int64>(i);
			}

			return data;
		}();

		// 33!! = 6,332,659,870,762,850,625 is the largest double factorial that fits in Int64; 34!! would overflow.
		// OneOffsettedDoubleFactorials is sized 34+1 to additionally hold (-1)!! at index 0.
		inline constexpr auto DoubleFactorials = []() constexpr
		{
			std::array<Int64, 34> data = {};

			for (std::size_t i = 0; i < data.size(); i++)
			{
				data[i] = CalculateDoubleFactorial<Int64>(static_cast<Int64>(i));
			}

			return data;
		}();

		inline constexpr auto OneOffsettedDoubleFactorials = []() constexpr
		{
			std::array<Int64, 34 + 1> data = {};

			data[-1 + 1] = 1;  // we know that (-1)!! is 1
			for (std::size_t i = 0; i < data.size() - 1; i++)
			{
				data[i + 1] = CalculateDoubleFactorial<Int64>(static_cast<Int64>(i));
			}

			return data;
		}();
	}

	/// <summary>
	/// Returns n! for n in [0, 20] from a compile-time table.
	/// </summary>
	/// <remarks>
	/// 20! is the largest factorial that fits in Int64, so the table does not extend past n = 20.
	/// </remarks>
	constexpr SEC_FORCE_INLINE Int64 Factorial(const Int64 i) noexcept
	{
		assert(i >= 0 && i <= 20);
		return Detail::Factorial::Factorials[i];
	}

	/// <summary>
	/// Returns n!! for n in [-1, 33] from a compile-time table.
	/// </summary>
	/// <remarks>
	/// Only (-1)!! is tabulated on the negative side; for other negative odd integers use CalculateDoubleFactorial.
	/// 33!! is the largest double factorial that fits in Int64.
	/// </remarks>
	constexpr SEC_FORCE_INLINE Int64 DoubleFactorial(const Int64 i) noexcept
	{
		assert(i >= -1 && i <= 33);
		return Detail::Factorial::OneOffsettedDoubleFactorials[i + 1];
	}
}
