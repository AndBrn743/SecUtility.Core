// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Math/ContinuedFraction.hpp>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Math/Special/Digamma.hpp>
#include <SecUtility/Math/Special/Factorial.hpp>
#include <cassert>
#include <limits>

#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
#define SEC_MATH_CONDITIONAL_CONSTEXPR constexpr
#else
#define SEC_MATH_CONDITIONAL_CONSTEXPR
#endif


namespace SecUtility::Math
{
	namespace Detail::ExpIntegral
	{

	}

	template <std::size_t N, typename T>
	SEC_MATH_CONDITIONAL_CONSTEXPR T ExpIntegral(const T x) noexcept
	{
		// N should never be very large, but static_assert cost nothing anyway
		static_assert(N <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
		assert(!(x < 0 || (x == 0 && (N == 0 || N == 1))) && "Bad arguments in ExpIntegral");

		if constexpr (N == 0)
		{
			return Exp(-x) / x;
		}

		if (x == 0)
		{
			return T{1} / (N - 1);
		}

		if (x > 0.8)
		{
			const auto cf = ContinuedFraction<T>([](const auto n) { return -(n + 1) * (static_cast<int>(N) + n); },
			                                     [x](const auto n) { return x + static_cast<int>(N) + 2 * (n + 1); });
			return Exp(-x) / (x + N + cf);
		}
		else
		{
			T result = (Pow(x, N - 1) / Factorial(N - 1)) * (-Log(x) + Math::Digamma<T>(N));

			T powerOfMinusXToMOverFactorialOfM = 1;
			for (int m = 0; m < 200; m++)
			{
				if (m + 1 != N)
				{
					const T term = powerOfMinusXToMOverFactorialOfM / (m + 1 - T{N});
					result -= term;

					if (Abs(term) < Abs(result) * std::numeric_limits<T>::epsilon())
					{
						return result;
					}
				}

				powerOfMinusXToMOverFactorialOfM *= -x / (m + 1);  // Pow(-x, m) / Factorial(m) of the next iteration
			}
		}

		return std::numeric_limits<double>::signaling_NaN();
	}
}

#undef SEC_MATH_CONDITIONAL_CONSTEXPR
