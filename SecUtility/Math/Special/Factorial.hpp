// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 Andy Brown

#pragma once

#include <SecUtility/Raw/Int.hpp>
#include <array>
#include <cassert>


namespace SecUtility::Math
{
	/// <summary>
	/// Calculate and returns the factorial of given non-negative number with recursive function calls
	/// </summary>
	template <typename T>
	constexpr T CalculateFactorial(const T n)  // NOLINT(*-no-recursion)
	{
		return n <= 1 ? 1 : n * CalculateFactorial(n - 1);
	}

	/// <summary>
	/// Calculate and returns the factorial of given number with recursive function calls.
	/// The number can either be positive or negative odd number
	/// </summary>
	template <typename T>
	constexpr T CalculateDoubleFactorial(const T n)  // NOLINT(*-no-recursion)
	{
		if (n < 0)
		{
			return CalculateDoubleFactorial(n + 2) / (n + 2);
		}

		return n <= 1 ? 1 : n * CalculateDoubleFactorial(n - 2);
	}

	namespace Detail::Factorial
	{
		inline constexpr auto Factorials = []() constexpr
		{
			std::array<Int64, 21> data = {};

			for (int i = 0; i < 21; i++)
			{
				data[i] = CalculateFactorial<Int64>(i);
			}

			return data;
		}();

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
	/// Returns factorial of number that's within the range of [0, 20].
	/// This method uses compile-time generated table and does not perform actual calculation.
	/// </summary>
	constexpr SEC_FORCE_INLINE Int64 Factorial(const Int64 i) noexcept
	{
		assert(i >= 0 && i <= 20);
		return Detail::Factorial::Factorials[i];
	}

	/// <summary>
	/// Returns double factorial of number that's within the range of [-1, 33].
	/// For the double factorials of negative odd number, use <c>CalculateDoubleFactorial</c> method instead.
	/// This method uses compile-time generated table and does not perform actual calculation.
	/// </summary>
	constexpr Int64 DoubleFactorial(const Int64 i) noexcept
	{
		assert(i >= -1 && i <= 33);
		return Detail::Factorial::OneOffsettedDoubleFactorials[i + 1];
	}
}
