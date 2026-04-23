// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Math/Constant.hpp>
#include <SecUtility/Math/Core.hpp>
#include <limits>


#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
#define SEC_MATH_CONDITIONAL_CONSTEXPR constexpr
#else
#define SEC_MATH_CONDITIONAL_CONSTEXPR
#endif

namespace SecUtility::Math
{
	template <typename T>
	SEC_MATH_CONDITIONAL_CONSTEXPR T Digamma(T x) noexcept
	{
		// Handle poles (non-positive integers)
		if (x <= 0)
		{
			const T nearest = Round(x);
			if (Abs(x - nearest) < 1e-14)
			{
				return std::numeric_limits<T>::quiet_NaN();
			}
		}

		// Use reflection for negative and small x
		if (x < 0.5)
		{
			return Digamma(1 - x) - Constant::Pi<T> / Tan(Constant::Pi<T> * x);
		}

		// Recurrence to large x
		T result = 0;

		while (x < 10)
		{
			result -= 1 / x;
			x += 1;
		}

		// Apply asymptotic expansion
		const T inv = 1 / x;
		const T inv2 = inv * inv;

		const T series =
		        Log(x) - 0.5 * inv - inv2 * (1. / 12. - inv2 * (1. / 120. - inv2 * (1. / 252. - inv2 * (1. / 240.))));

		return result + series;
	}
}

#undef SEC_MATH_CONDITIONAL_CONSTEXPR
