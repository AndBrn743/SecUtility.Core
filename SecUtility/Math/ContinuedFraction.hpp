// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>
#include <limits>


namespace SecUtility::Math
{
	/**
	 * Evaluates a continued fraction using the recurrence:
	 *   h_n = a_n * h_{n-1} + h_{n-2}
	 *   k_n = a_n * k_{n-1} + k_{n-2}
	 *
	 * @param nestedNumerators functor which returns a_i (partial numerators, i >= 0)
	 * @param nestedDenominators functor which returns b_i (partial denominators, i >= 0)
	 * @param tolerance   convergence threshold
	 * @param maxIteration    maximum number of iterations before giving up
	 *
	 * The continued fraction evaluated is:
	 *
	 *          a_0
	 *   -----------------
	 *             a_1
	 *    b_0 + ---------
	 *          b_1 + ...
	 */
	template <typename Scalar, typename NestedNumeratorFn, typename NestedDenominatorFn>
	constexpr Scalar ContinuedFraction(NestedNumeratorFn nestedNumerators,
	                                   NestedDenominatorFn nestedDenominators,
	                                   const std::type_identity_t<Scalar> tolerance =
	                                           1000 * std::numeric_limits<std::type_identity_t<Scalar>>::epsilon(),
	                                   const int maxIteration = 200)  //
	        noexcept(noexcept(nestedNumerators(1)) && noexcept(nestedDenominators(1)))
	{
		Scalar previousNumerator = 1;
		Scalar currentNumerator = 0;
		Scalar previousDenominator = 0;
		Scalar currentDenominator = 1;

		Scalar previousValue = std::numeric_limits<Scalar>::infinity();

		for (int i = 0; i <= maxIteration; i++)
		{
			const auto nestedNumerator = static_cast<Scalar>(nestedNumerators(i));
			const auto nestedDenominator = static_cast<Scalar>(nestedDenominators(i));

			// clang-format off
			const Scalar updatedNumerator = nestedDenominator * currentNumerator + nestedNumerator * previousNumerator;
			const Scalar updatedDenominator = nestedDenominator * currentDenominator + nestedNumerator * previousDenominator;
			// clang-format on

			if (updatedDenominator == 0)
			{
				return std::numeric_limits<Scalar>::infinity();
			}

			const Scalar value = updatedNumerator / updatedDenominator;

			if (Abs(value - previousValue) < tolerance)
			{
				return value;
			}

			previousNumerator = currentNumerator;
			currentNumerator = updatedNumerator;
			previousDenominator = currentDenominator;
			currentDenominator = updatedDenominator;
			previousValue = value;
		}

		return std::numeric_limits<Scalar>::signaling_NaN();
	}


	/**
	 * Evaluates a continued fraction using the recurrence:
	 * The continued fraction evaluated is:
	 *
	 *           1
	 *   -----------------
	 *              1
	 *    c_0 + ---------
	 *          c_1 + ...
	 */
	template <typename Scalar, typename CoefficientFn>
	constexpr Scalar SimpleContinuedFraction(CoefficientFn coefficients,
	                                         const std::type_identity_t<Scalar> tolerance =
	                                                 std::type_identity_t<Scalar>{1000}
	                                                 * std::numeric_limits<std::type_identity_t<Scalar>>::epsilon(),
	                                         const int maxIteration = 200)  //
	        noexcept(noexcept(coefficients(0)))
	{
		return ContinuedFraction<Scalar>([](const auto) { return 1; }, coefficients, tolerance, maxIteration);
	}
}
