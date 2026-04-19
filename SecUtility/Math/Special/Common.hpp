// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 Andy Brown

#pragma once

#include <SecUtility/Macro/ConstevalIf.hpp>
#include <SecUtility/Macro/ForceInline.hpp>
#include <SecUtility/Math/Core.hpp>
#include <numeric>

#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
#include <gcem.hpp>
#define SEC_MATH_CONDITIONAL_CONSTEXPR constexpr
#else
#define SEC_MATH_CONDITIONAL_CONSTEXPR
#endif

#if defined(SEC_MATH_CORE_EXTENSION)
#if __has_include(SEC_MATH_CORE_EXTENSION)
#include SEC_MATH_CORE_EXTENSION
#endif
#endif

#include <cassert>
#include <cmath>
#include <utility>


namespace SecUtility::Math
{
#define SEC_EXPORT_RUNTIME_ONLY_MATH_FUNCTION(SEC_NAME, C_NAME)                                                        \
	template <typename... Args>                                                                                        \
	SEC_FORCE_INLINE auto SEC_NAME(Args&&... args) noexcept(noexcept(std::C_NAME(std::forward<Args>(args)...)))        \
	        -> decltype(std::C_NAME(std::forward<Args>(args)...))                                                      \
	{                                                                                                                  \
		return std::C_NAME(std::forward<Args>(args)...);                                                               \
	}

#define SEC_EXPORT_CONSTEXPR_MATH_FUNCTION_WITH_FALLBACK(SEC_NAME, C_NAME)                                             \
	template <typename... Args>                                                                                        \
	constexpr SEC_FORCE_INLINE auto SEC_NAME(Args&&... args) noexcept(                                                 \
	        noexcept(std::C_NAME(std::forward<Args>(args)...))) -> decltype(std::C_NAME(std::forward<Args>(args)...))  \
	{                                                                                                                  \
		SEC_IF_CONSTEVAL                                                                                               \
		{                                                                                                              \
			return gcem::C_NAME(std::forward<Args>(args)...);                                                          \
		}                                                                                                              \
		else                                                                                                           \
		{                                                                                                              \
			return std::C_NAME(std::forward<Args>(args)...);                                                           \
		}                                                                                                              \
	}

#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
#define SEC_EXPORT_MATH_FUNCTION SEC_EXPORT_CONSTEXPR_MATH_FUNCTION_WITH_FALLBACK
#else
#define SEC_EXPORT_MATH_FUNCTION SEC_EXPORT_RUNTIME_ONLY_MATH_FUNCTION
#endif

	SEC_EXPORT_MATH_FUNCTION(Erf, erf)
	SEC_EXPORT_MATH_FUNCTION(Gamma, tgamma)
	SEC_EXPORT_MATH_FUNCTION(LogGamma, lgamma)

	SEC_EXPORT_MATH_FUNCTION(GreatestCommonDivisor, gcd)
	SEC_EXPORT_MATH_FUNCTION(LeastCommonMultiple, lcm)

#undef SEC_EXPORT_RUNTIME_ONLY_MATH_FUNCTION
#undef SEC_EXPORT_CONSTEXPR_MATH_FUNCTION_WITH_FALLBACK
#undef SEC_EXPORT_MATH_FUNCTION

	template <typename Arg>
	SEC_MATH_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE auto Erfc(const Arg arg) noexcept(noexcept(std::erfc(arg)))
	        -> decltype(std::erfc(arg))
	{
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
		SEC_IF_CONSTEVAL
		{
			return 1 - gcem::erf(arg);
		}
#endif

		return std::erfc(arg);
	}

	template <typename Scalar>
	SEC_MATH_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE Scalar
	LogBeta(const Scalar x, const Scalar y) noexcept(noexcept(LogGamma(x) + LogGamma(y) - LogGamma(x + y)))
	{
		return LogGamma(x) + LogGamma(y) - LogGamma(x + y);
	}

	template <typename Scalar>
	SEC_MATH_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE Scalar Beta(const Scalar x,
	                                                            const Scalar y) noexcept(noexcept(Exp(LogBeta(x, y))))
	{
		return Exp(LogBeta(x, y));
	}

	template <typename Scalar>
	constexpr Scalar Logistic(const Scalar x)
	{
		return 1 / (1 + Exp(-x));
	}


	template <typename Integer>
	constexpr Integer BinomialCoefficient(const Integer n, Integer k) noexcept
	{
		static_assert(std::is_integral_v<std::decay_t<Integer>>);

		if (k > n)
		{
			return 0;
		}
		if (k == 0 || k == n)
		{
			return 1;
		}
		if (k > n - k)
		{
			k = n - k;
		}

		Integer result = 1;
		for (Integer i = 1; i <= k; i++)
		{
			result *= n - i + 1;
			result /= i;
		}
		return result;
	}
}

#undef SEC_MATH_CONDITIONAL_CONSTEXPR
