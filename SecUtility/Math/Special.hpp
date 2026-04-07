//
// Created by Andy on 3/7/2026.
//

#pragma once

#include "Core.hpp"
#include <SecUtility/Macro/ConstevalIf.hpp>
#include <SecUtility/Macro/ForceInline.hpp>
#include <SecUtility/Raw/Int.hpp>
#include <numeric>

#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
#include <gcem.hpp>
#endif

#if defined(SEC_MATH_CORE_EXTENSION)
#if __has_include(SEC_MATH_CORE_EXTENSION)
#include SEC_MATH_CORE_EXTENSION
#endif
#endif

#include <array>
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
		SEC_IF_CONSTEVAL return gcem::C_NAME(std::forward<Args>(args)...);                                             \
		else return std::C_NAME(std::forward<Args>(args)...);                                                          \
	}

#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
#define SEC_EXPORT_MATH_FUNCTION SEC_EXPORT_CONSTEXPR_MATH_FUNCTION_WITH_FALLBACK
#else
#define SEC_EXPORT_MATH_FUNCTION SEC_EXPORT_RUNTIME_ONLY_MATH_FUNCTION
#endif

	SEC_EXPORT_MATH_FUNCTION(Erf, erf)
	SEC_EXPORT_MATH_FUNCTION(Gamma, tgamma)
	SEC_EXPORT_MATH_FUNCTION(LogGamma, lgamma)
	SEC_EXPORT_MATH_FUNCTION(Beta, beta)

	SEC_EXPORT_MATH_FUNCTION(GreatestCommonDivisor, gcd)
	SEC_EXPORT_MATH_FUNCTION(LeastCommonMultiple, lcm)

#undef SEC_EXPORT_RUNTIME_ONLY_MATH_FUNCTION
#undef SEC_EXPORT_CONSTEXPR_MATH_FUNCTION_WITH_FALLBACK
#undef SEC_EXPORT_MATH_FUNCTION

	template <typename Arg>
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
	constexpr
#endif
	        SEC_FORCE_INLINE auto Erfc(const Arg arg) noexcept(noexcept(std::erfc(arg))) -> decltype(std::erfc(arg))
	{
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
		SEC_IF_CONSTEVAL
		{
			return 1 - gcem::erf(arg);
		}
#endif

		return std::erfc(arg);
	}

	template <typename... Args>
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
	constexpr
#endif
	        SEC_FORCE_INLINE auto LogBeta(Args&&... args) noexcept(
	                noexcept(std::log(std::beta(std::forward<Args>(args)...))))
	                -> decltype(std::log(std::beta(std::forward<Args>(args)...)))
	{
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
		SEC_IF_CONSTEVAL
		{
			return gcem::lbeta(std::forward<Args>(args)...);
		}
#endif

		return std::log(std::beta(std::forward<Args>(args)...));
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

	//----------------------------------------------------------------------------------------------------------------//

	namespace Advanced
	{
		/// Calculate upper incomplete gamma using series expansion, suitable for small x
		template <typename T>
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
		constexpr
#endif
		        T GammaUpperIncompleteWithSeriesExpansion(const T a, const T x) noexcept
		{
			if (x == 0)
			{
				return Gamma(a);  // Γ(a, 0) = Γ(a)
			}

			if (x < 0 || a <= 0)
			{
				return std::numeric_limits<T>::signaling_NaN();
			}

			// Compute series expansion for lower incomplete gamma
			T term = 1.0 / a;
			T sum = term;
			for (int n = 1; n < 100; n++)
			{
				term *= x / (a + n);
				sum += term;
				if (Abs(term) < std::numeric_limits<T>::epsilon() * Abs(sum))
				{
					break;
				}
			}
			const T lowerGamma = sum * Exp(-x + a * Log(x));
			return Gamma(a) - lowerGamma;  // Use the relationship: Γ(a, x) = Γ(a) - γ(a, x)
		}

		/// Calculate upper incomplete gamma using continued fraction, suitable for large x
		template <typename T>
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
		constexpr
#endif
		        T GammaUpperIncompleteWithContinuedFraction(const T s, const T x) noexcept
		{
			// NOTE: Naming convention violated so it fits the equation better
			const T a = 1 - s;
			T b = a + x + 1;
			T f = 1.0 / b;
			T C = f;
			T D = 0;

			for (int n = 1; n < 100; ++n)
			{
				const T an = n * (s - n);
				b += 2;
				D = an * D + b;

				if (Abs(D) < std::numeric_limits<T>::epsilon())
				{
					D = std::numeric_limits<T>::epsilon();
				}

				C = b + an / C;

				if (Abs(C) < std::numeric_limits<T>::epsilon())
				{
					C = std::numeric_limits<T>::epsilon();
				}

				D = 1.0 / D;
				const T delta = C * D;
				f *= delta;

				if (Abs(delta - 1.0) < 1e-12)
				{
					break;
				}
			}

			return Pow(x, s) * Exp(-x) * f;
		}

		/// Calculate upper incomplete gamma using series expansion, suitable for x >> s
		template <typename T>
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
		constexpr
#endif
		        T GammaUpperIncompleteWithAsymptoticExpansion(const T s, const T x) noexcept
		{
			constexpr int maxIter = 20;  // Typically, ~10 iterations are enough for good accuracy
			T sum = 1.0;
			T term = 1.0;

			for (int n = 1; n < maxIter; n++)
			{
				term *= (s - n) / x;  // Compute next term efficiently
				sum += term;

				if (Abs(term) < 1e-12)
				{
					break;  // Convergence check
				}
			}

			return Pow(x, s - 1) * Exp(-x) * sum;
		}
	}

	template <typename T>
#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
	constexpr
#endif
	        SEC_FORCE_INLINE T GammaUpperIncomplete(const T a, const T x) noexcept
	{
		if (x > 10 * a)
		{
			return GammaUpperIncompleteWithAsymptoticExpansion(a, x);
		}
		else  // if (x < a + 1)
		{
			return GammaUpperIncompleteWithSeriesExpansion(a, x);
		}
		// else
		// {
		// 	return GammaUpperIncompleteWithContinuedFraction(a, x);
		// }
	}

	//----------------------------------------------------------------------------------------------------------------//

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

	namespace Detail
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

	/// <summary>
	/// Returns factorial of number that's within the range of [0, 20].
	/// This method uses compile-time generated table and does not perform actual calculation.
	/// </summary>
	constexpr SEC_FORCE_INLINE Int64 Factorial(const Int64 i) noexcept
	{
		assert(i >= 0 && i <= 20);
		return Detail::Factorials[i];
	}

	/// <summary>
	/// Returns double factorial of number that's within the range of [-1, 33].
	/// For the double factorials of negative odd number, use <c>CalculateDoubleFactorial</c> method instead.
	/// This method uses compile-time generated table and does not perform actual calculation.
	/// </summary>
	constexpr Int64 DoubleFactorial(const Int64 i) noexcept
	{
		assert(i >= -1 && i <= 33);
		return Detail::OneOffsettedDoubleFactorials[i + 1];
	}

	/// <summary>
	/// Returns the Gamma function value of arg which is a positive half integer
	/// </summary>
	/// <remarks>
	/// For half integers be -62.5, -61.5, ..., -0.5, 0.5, 1.5, ..., 63.5 and compile-time computed value will be
	/// returned, otherwise a runtime calculation will be performed
	/// </remarks>
	template <typename Scalar>
	constexpr SEC_FORCE_INLINE Scalar GammaOfHalfInteger(const Scalar halfInteger) noexcept
	{
		const auto floor = static_cast<int>(Floor(halfInteger));

		if (floor >= 0 && static_cast<std::size_t>(floor) < Detail::GammaOfPositiveHalfIntegers<Scalar>.size())
		{
			return Detail::GammaOfPositiveHalfIntegers<Scalar>[floor];
		}

		if (floor < 0 && static_cast<std::size_t>(-floor - 1) < Detail::GammaOfNegativeHalfIntegers<Scalar>.size())
		{
			return Detail::GammaOfNegativeHalfIntegers<Scalar>[-floor - 1];
		}

		return Gamma(halfInteger);
	}
}
