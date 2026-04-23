// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 Andy Brown

#pragma once

#include <SecUtility/Collection/IndexAccessor.hpp>
#include <SecUtility/Math/HornerPolynomial.hpp>
#include <SecUtility/Math/Special/Gamma.hpp>
#include <array>
#include <cassert>
#include <vector>


#if defined(SEC_IF_CONSTEVAL) && __has_include(<gcem.hpp>)
#define SEC_MATH_CONDITIONAL_CONSTEXPR constexpr
#else
#define SEC_MATH_CONDITIONAL_CONSTEXPR
#endif

namespace SecUtility::Math
{
	namespace Detail::Boys
	{
		/// <summary>
		/// Calculate Boys function with expansion, suitable when x is not large, e.g., < 12.
		/// <m>F_m(x) = \frac{1}{2} e^{-x} \sum_{k=0}^{\infty} \frac{x^k}{\prod_{j=0}^{k} (m + j + \tfrac{1}{2})}</m>
		/// </summary>
		template <typename Scalar>
		SEC_MATH_CONDITIONAL_CONSTEXPR Scalar HighPrecisionBoysWithExpansion(const int m,
		                                                                     const Scalar x,
		                                                                     const Scalar tolerance = 1e-16) noexcept
		{
			Scalar denominator = m + Scalar{0.5};
			Scalar term = 1 / denominator;
			Scalar partialSum = term;
			for (int k = 0; k < 150; k++)
			{
				denominator += 1;
				term *= x / denominator;
				partialSum += term;
				if (term < tolerance * partialSum)
				{
					return Scalar{0.5} * Exp(-x) * partialSum;
				}
			}

			return std::numeric_limits<Scalar>::signaling_NaN();
		}

		/// <summary>
		/// Calculate Boys function using its asymptotic behavior when arg is not small, e.g., > 12.
		/// </summary>
		template <typename Scalar>
		SEC_MATH_CONDITIONAL_CONSTEXPR Scalar HighPrecisionBoysWithAsymptotic(const int m, const Scalar x) noexcept
		{
			if (x <= 0)
			{
				return Scalar{1} / (2 * m + 1);
			}

			const Scalar a = m + Scalar{0.5};

			const Scalar logGammaA = LogGamma(a);

			// Compute prefactor in log-space:
			// (1/2) * Γ(a) * x^{-a}
			const Scalar logOfTwoTimesThePrefactor = logGammaA - a * Log(x);

			// Regularized Q(a, x) = Γ(a,x) / Γ(a)
			const Scalar q = Gamma::GammaQ(a, x, logGammaA);

			// Final result
			return Scalar{0.5} * Exp(logOfTwoTimesThePrefactor) * (Scalar{1} - q);
		}

		/// <summary>
		/// Calculate Boys function using its asymptotic behavior when arg is large, e.g., > 27.
		/// </summary>
		template <typename Scalar>
		constexpr SEC_FORCE_INLINE Scalar BoysWithAsymptotic(const int n, const Scalar x) noexcept
		{
			return GammaOfHalfInteger<Scalar>(n + 0.5) / (2 * PowInt(x, static_cast<unsigned int>(n)) * Sqrt(x));
		}

		/// <summary>
		/// Calculate and returns the Boys function,
		/// F<sub>n</sub>(a) = ʃ<sub>0</sub><sup>1</sup> exp(-ax<sup>2</sup>) x<sup>2n</sup> dx, with high precision.
		/// The typical use case is for generate lookup tables and other things that require high accuracy.
		/// </summary>
		template <typename Scalar>
		SEC_MATH_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE Scalar
		HighPrecisionBoys(const int m, const Scalar x, const Scalar tolerance = 1e-16) noexcept
		{
			return x <= 12 ? Boys::HighPrecisionBoysWithExpansion(m, x, tolerance)
			               : Boys::HighPrecisionBoysWithAsymptotic(m, x);
		}

		inline constexpr auto HighPrecisionBoysFn = [](const int m, const auto x)
		                                                    SEC_MATH_CONDITIONAL_CONSTEXPR noexcept
		{ return HighPrecisionBoys(m, x); };

		/// <summary>
		/// Use the downward recursion to calculate Boys functions
		/// </summary>
		template <typename BidirectionalIterator, typename Scalar, typename BoysFn>
		SEC_MATH_CONDITIONAL_CONSTEXPR void PopulateContainerWithBoysFunctionValuesFromHighest(
		        const BidirectionalIterator begin, const BidirectionalIterator end, const Scalar x, BoysFn boys)
		{
			Scalar lastBoys = std::numeric_limits<Scalar>::signaling_NaN();

			auto iterator = end;

			int n = static_cast<int>(std::distance(begin, end)) - 1;

			{
				iterator--;

				// always use `HighPrecisionBoys` to populate the table
				*iterator = boys(n, x);

				n--;
				lastBoys = *iterator;
			}

			const Scalar expMinusX = Exp(-x);

			while (n != -1)
			{
				iterator--;
				*iterator = (2 * x * lastBoys + expMinusX) / (2 * n + 1);

				n--;
				lastBoys = *iterator;
			}
		}

		template <int MaxOrder, auto MaxArg, int GridDensity>
		SEC_MATH_CONDITIONAL_CONSTEXPR auto CreateBoysTable() noexcept
		{
			using Scalar = std::conditional_t<std::is_floating_point_v<decltype(MaxArg)>, decltype(MaxArg), double>;
			constexpr auto Count = Ceil<std::size_t>(MaxArg * GridDensity);
			std::array<std::array<Scalar, MaxOrder + 1>, Count> fs{};

			if constexpr (Count == 0)
			{
				return fs;
			}

			constexpr Scalar Delta = Scalar{1} / GridDensity;

			for (std::size_t i = 0; i < Count; i++)
			{
				const Scalar x = Delta * (static_cast<Scalar>(i) + 0.5);
				PopulateContainerWithBoysFunctionValuesFromHighest(fs[i].begin(), fs[i].end(), x, HighPrecisionBoysFn);
			}

			return fs;
		}

		/// <summary>
		/// Use the upward recursion to calculate Boys functions
		/// </summary>
		template <typename ForwardIterator, typename Scalar, typename BoysFn>
		SEC_MATH_CONDITIONAL_CONSTEXPR void PopulateContainerWithBoysFunctionValuesFromLowest(
		        const ForwardIterator begin, const ForwardIterator end, const Scalar x, BoysFn boys)
		{
			Scalar lastBoys = std::numeric_limits<Scalar>::signaling_NaN();

			auto iterator = begin;

			{
				*iterator = boys(0, x);

				lastBoys = *iterator;
				iterator++;
			}

			if (iterator == end)
			{
				return;
			}

			const Scalar oneOverTwoX = Scalar{0.5} / x;
			const Scalar expMinusX = Exp(-x);

			for (int n = 1; iterator != end; n++, iterator++)  // NOTE: `n` starts from 1
			{
				*iterator = oneOverTwoX * ((2 * n - 1) * lastBoys - expMinusX);

				if (*iterator > lastBoys)  // diagnose me!!
				{
					*iterator = boys(n, x);
				}

				lastBoys = *iterator;
			}
		}

		static constexpr int MaxTabulatedBoyOrder =
#if defined(SEC_MATH_MAX_TABULATED_BOYS_ORDER)
		        SEC_MATH_MAX_TABULATED_BOYS_ORDER;
#else
		        22;
#endif

		static constexpr
#if __cplusplus >= 202002L
		        double
#else
		        int
#endif
		                MaxTabulatedBoyArg =
#if defined(SEC_MATH_MAX_TABULATED_BOYS_ARG)
		                        SEC_MATH_MAX_TABULATED_BOYS_ARG;
#else
		                        12;
#endif

		static constexpr int BoyTableDensity =
#if defined(SEC_MATH_MAX_TABULATED_BOYS_DENSITY)
		        SEC_MATH_MAX_TABULATED_BOYS_DENSITY;
#else
		        5;
#endif

		static_assert(MaxTabulatedBoyOrder >= 1);
		static_assert(MaxTabulatedBoyArg > 0);
		static_assert(BoyTableDensity >= 1);

		inline
#if defined(SEC_IF_CONSTEVAL)
		        constexpr
#else
		        const
#endif
		        auto BoysTable = CreateBoysTable<MaxTabulatedBoyOrder, MaxTabulatedBoyArg, BoyTableDensity>();
	}


	/// <summary>
	/// Calculate and returns the Boys function with high precision (Δ < 1e-14 in our tests).
	/// F<sub>n</sub>(a) = ʃ<sub>0</sub><sup>1</sup> exp(-ax<sup>2</sup>) x<sup>2n</sup> dx
	/// </summary>
	/// <remarks>
	/// This functor is designed for compile-time use. For better runtime performance, please consider <c>Math::Boys</c>
	/// instead. This is made a functor to support compile-time dependency injection downstream.
	/// </remarks>
	constexpr auto& HighPrecisionBoys = Detail::Boys::HighPrecisionBoysFn;


	/// <summary>
	/// Calculate and returns the Boys function with default accuracy (Δ < 1e-12 in our tests).
	/// F<sub>n</sub>(a) = ʃ<sub>0</sub><sup>1</sup> exp(-ax<sup>2</sup>) x<sup>2n</sup> dx
	/// </summary>
	/// <remarks>
	/// This is made a functor to support compile-time dependency injection downstream.
	/// </remarks>
	constexpr auto Boys = [](const int n, const auto a) SEC_MATH_CONDITIONAL_CONSTEXPR noexcept
	{
		using Scalar = std::decay_t<decltype(a)>;

		if (a < 1e-8)
		{
			return Scalar{1} / (2 * n + 1) - a / (2 * n + 3);
		}

		if (n == 0)
		{
			constexpr Scalar HalfOfSqrtOfPi = Constant::SqrtOfPi<Scalar> / 2;
			const Scalar sqrtOfA = Sqrt(a);
			return HalfOfSqrtOfPi * Erf(sqrtOfA) / sqrtOfA;
		}

		if (a < Detail::Boys::MaxTabulatedBoyArg)
		{
			constexpr auto HornerTermCount = 8;

			if (n + HornerTermCount - 1 >= Detail::Boys::MaxTabulatedBoyOrder)
			{
				return Detail::Boys::HighPrecisionBoys(n, a, 1e-10);
			}

			const auto gridIndex = static_cast<std::size_t>(a * Detail::Boys::BoyTableDensity);
			const auto delta = (gridIndex + 0.5) / Detail::Boys::BoyTableDensity - a;

			return UnrolledHornerTaylorPolynomial<Scalar, HornerTermCount>(
			        MakeIndexAccessor(Detail::Boys::BoysTable[gridIndex].cbegin() + n), delta);
		}

		if (a > 27)
		{
			return Detail::Boys::BoysWithAsymptotic(n, a);
		}

		return Detail::Boys::HighPrecisionBoysWithAsymptotic(n, a);
	};


	/// <summary>
	/// Use either upward or downward recursion to calculate Boys functions
	/// </summary>
	template <typename ForwardIterator, typename Scalar>
	SEC_MATH_CONDITIONAL_CONSTEXPR SEC_FORCE_INLINE void PopulateContainerWithBoysFunctionValues(
	        const ForwardIterator begin, const ForwardIterator end, const Scalar x)
	{
		if (Abs(x) < 1e-8)
		{
			auto iterator = begin;
			for (int n = 0; iterator != end; n++, iterator++)
			{
				*iterator = Scalar{1} / (2 * n + 1);
			}
		}
		else if (x > 10)
		{
			Detail::Boys::PopulateContainerWithBoysFunctionValuesFromLowest(begin, end, x, Boys);
		}
		else
		{
			Detail::Boys::PopulateContainerWithBoysFunctionValuesFromHighest(begin, end, x, Boys);
		}
	}

	template <typename Integer, typename Scalar>
	std::vector<Scalar> BoysList(const Integer n, const Scalar x)
	{
		assert(n >= 0);
		std::vector<Scalar> result(n + 1);
		PopulateContainerWithBoysFunctionValues(result.begin(), result.end(), x);
		return result;
	}

	template <std::size_t N, typename Scalar>
	std::array<Scalar, N + 1> BoysArray(const Scalar x) noexcept
	{
		std::array<Scalar, N + 1> result{};
		PopulateContainerWithBoysFunctionValues(result.begin(), result.end(), x);
		return result;
	}
}

#undef SEC_MATH_CONDITIONAL_CONSTEXPR
