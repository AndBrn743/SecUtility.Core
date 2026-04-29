// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 Andy Brown

#pragma once

#include <SecUtility/Math/ContinuedFraction.hpp>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Math/Special/Common.hpp>
#include <SecUtility/Math/Special/ExpIntegral.hpp>
#include <SecUtility/Math/Special/Factorial.hpp>

#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
#define SEC_MATH_CONDITIONAL_CONSTEXPR constexpr
#else
#define SEC_MATH_CONDITIONAL_CONSTEXPR
#endif


namespace SecUtility::Math
{
	namespace Detail::Gamma
	{
		template <typename T>
		SEC_MATH_CONDITIONAL_CONSTEXPR T GammaQ_ContinuedFraction(const T a, const T x, const T logGammaA) noexcept
		{
			const auto cf = ContinuedFraction<T>([a](const int i) { return i == 0 ? 1 : i * (a - i); },
			                                     [a, x](const int i) { return x - a + 2 * i + 1; },
			                                     1000 * std::numeric_limits<T>::epsilon());
			return Exp(-x + a * Log(x) - logGammaA) * cf;
		}

		template <typename T>
		SEC_MATH_CONDITIONAL_CONSTEXPR T GammaP_Series(const T a, const T x, const T logGammaA) noexcept
		{
			constexpr T eps = std::numeric_limits<T>::epsilon();

			T term = 1 / a;
			T sum = term;

			for (int n = 1; n < 200; ++n)
			{
				term *= x / (a + n);
				sum += term;

				if (Abs(term) < Abs(sum) * eps)
				{
					break;
				}
			}

			return sum * Exp(-x + a * Log(x) - logGammaA);
		}

		template <typename T>
		SEC_MATH_CONDITIONAL_CONSTEXPR T GammaQ_Asymptotic(const T a, const T x, const T logGammaA) noexcept
		{
			constexpr int maxIter = 20;

			T sum = 1;
			T term = 1;

			for (int n = 1; n < maxIter; n++)
			{
				term *= (a - n) / x;
				sum += term;

				if (Abs(term) < 1e-12)
				{
					break;
				}
			}

			return sum * Exp((a - 1) * Log(x) - x - logGammaA);
		}

		template <typename T>
		SEC_MATH_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE T GammaQ(const T a, const T x, const T logGammaA) noexcept
		{
			if (x < 0 || a <= 0)
			{
				return std::numeric_limits<T>::quiet_NaN();
			}

			if (x == 0)
			{
				return T{1};
			}

			// --- asymptotic (large x) ---
			if (x > a + 50)
			{
				return Gamma::GammaQ_Asymptotic(a, x, logGammaA);
			}

			// --- small x: use P ---
			if (x < a + 1)
			{
				return 1 - Gamma::GammaP_Series(a, x, logGammaA);
			}

			// --- default: continued fraction ---
			return Gamma::GammaQ_ContinuedFraction(a, x, logGammaA);
		}

		template <typename T>
		SEC_MATH_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE T LogGammaQ(const T a, const T x, const T logGammaA) noexcept
		{
			if (x < 0 || a <= 0)
			{
				return std::numeric_limits<T>::quiet_NaN();
			}

			if (x == 0)
			{
				return T{0};
			}

			return Log(GammaQ(a, x, logGammaA));
		}

		template <typename Scalar>
		inline constexpr auto GammaOfPositiveHalfIntegers = []
		{
			std::array<Scalar, 64> result{};
			for (int i = 0; i < static_cast<int>(result.size()); i++)
			{
				// The equation used here can be found in Wikipedia, under "Particular values of the gamma function"
				result[i] = Constant::SqrtOfPi<Scalar>
				            * CalculateDoubleFactorial<Scalar>(static_cast<Scalar>(2 * i - 1)) / (UInt64{1} << i);
			}
			return result;
		}();

		template <typename Scalar>
		inline constexpr auto GammaOfNegativeHalfIntegers = []
		{
			std::array<Scalar, 63> result{};
			for (int i = 0; i < static_cast<int>(result.size()); i++)
			{
				// The equation used here can be found in Wikipedia, under "Particular values of the gamma function"
				result[i] = MinusOneToThePowerOf<Scalar>(i + 1) * Constant::SqrtOfPi<Scalar> * (UInt64{1} << (i + 1))
				            / CalculateDoubleFactorial<Scalar>(static_cast<Scalar>(2 * (i + 1) - 1));
			}
			return result;
		}();
	}

	template <typename T>
	SEC_MATH_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE T LogGammaQ(const T a, const T x) noexcept
	{
		return Detail::Gamma::LogGammaQ(a, x, LogGamma(a));
	}

	template <typename T>
	SEC_MATH_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE T GammaQ(const T a, const T x) noexcept
	{
		return Detail::Gamma::GammaQ(a, x, LogGamma(a));
	}


	template <typename T>
	SEC_MATH_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE T GammaUpperIncomplete(const T a, const T x) noexcept
	{
		if (x < 0 || a <= 0)
		{
			return std::numeric_limits<T>::signaling_NaN();
		}

		if (x == 0)
		{
			return Gamma(a);
		}

		if (a < 1e-6)
		{
			return Math::ExpIntegral<1>(x);
		}

		const auto logGammaA = LogGamma(a);
		return Exp(logGammaA) * Detail::Gamma::GammaQ(a, x, logGammaA);
	}

	/// <summary>
	/// Returns the Gamma function value of arg which is a positive half integer
	/// </summary>
	/// <remarks>
	/// If the half integers was -62.5, -61.5, ..., -0.5, 0.5, 1.5, ..., or 63.5, the compile-time computed value will
	/// be returned, otherwise a runtime calculation will be performed
	/// </remarks>
	template <typename Scalar>
	SEC_MATH_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE Scalar GammaOfHalfInteger(const Scalar halfInteger) noexcept
	{
		const auto floor = Floor<int>(halfInteger);

		if (floor >= 0 && static_cast<std::size_t>(floor) < Detail::Gamma::GammaOfPositiveHalfIntegers<Scalar>.size())
		{
			return Detail::Gamma::GammaOfPositiveHalfIntegers<Scalar>[floor];
		}

		if (floor < 0
		    && static_cast<std::size_t>(-floor - 1) < Detail::Gamma::GammaOfNegativeHalfIntegers<Scalar>.size())
		{
			return Detail::Gamma::GammaOfNegativeHalfIntegers<Scalar>[-floor - 1];
		}

		return Gamma(halfInteger);
	}
}

#undef SEC_MATH_CONDITIONAL_CONSTEXPR
