// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Macro/ForceInline.hpp>
#include <SecUtility/Meta/IntegerSequence.hpp>
#include <SecUtility/Math/Special/Factorial.hpp>

#include <array>
#include <cstddef>


namespace SecUtility::Math
{
	namespace Detail::HornerPolynomial
	{
		// power: ∑ c_k x^k
		// taylor: ∑ c_k x^k / k!

		template <typename Scalar, std::size_t N, typename CoefficientAccessor>
		constexpr Scalar HornerPowerPolynomial_Loop(const CoefficientAccessor& coefficientAccessor,
		                                            const Scalar delta) noexcept
		{
			static_assert(N > 0);
			Scalar result = 0;

			for (std::size_t reversedIndices = N; reversedIndices-- > 1;)
			{
				result += coefficientAccessor(reversedIndices);
				result *= delta;
			}

			result += coefficientAccessor(0);

			return result;
		}

		template <typename Scalar, std::size_t N, typename CoefficientAccessor, std::size_t... ReversedIndices>
		SEC_FORCE_INLINE constexpr Scalar HornerPowerPolynomial_Fold(const CoefficientAccessor& coefficientAccessor,
		                                                             const Scalar delta,
		                                                             std::index_sequence<ReversedIndices...>) noexcept
		{
			static_assert(N > 0);
			static_assert(sizeof...(ReversedIndices) == N);
			static_assert(((ReversedIndices < N) && ...));
			Scalar result = 0;
			// FMA-friendly: each step is c[k] + delta*result, which contracts to
			// fma(delta, result, c[k]) under -ffp-contract=on.  No per-step ternary
			// needed: the recurrence r_k = c[k] + delta*r_{k+1} is valid for all k.
			((result = coefficientAccessor(ReversedIndices) + delta * result), ...);
			return result;
		}

		template <typename Scalar, std::size_t N, typename CoefficientAccessor>
		constexpr Scalar HornerTaylorPolynomial_Loop(const CoefficientAccessor& coefficientAccessor,
		                                             const Scalar delta) noexcept
		{
			static_assert(!std::is_integral_v<Scalar>);
			static_assert(N > 0);
			Scalar result = 0;

			for (std::size_t reversedIndices = N; reversedIndices-- > 1;)
			{
				result += coefficientAccessor(reversedIndices);
				result *= delta * (Scalar{1} / reversedIndices);
			}

			result += coefficientAccessor(0);

			return result;
		}

		template <typename Scalar, std::size_t N, typename CoefficientAccessor, std::size_t... ReversedIndices>
		SEC_FORCE_INLINE constexpr Scalar HornerTaylorPolynomial_Fold(const CoefficientAccessor& coefficientAccessor,
		                                                              const Scalar delta,
		                                                              std::index_sequence<ReversedIndices...>) noexcept
		{
			static_assert(!std::is_integral_v<Scalar>);
			static_assert(N > 0);
			static_assert(sizeof...(ReversedIndices) == N);
			static_assert(((ReversedIndices < N) && ...));

			Scalar result = 0;
			// FMA-friendly: r_k = c[k] * (1/k!) + delta * r_{k+1}, which contracts to
			// fma(delta, result, c[k] * (1/k!)) under -ffp-contract=on.  The
			// c[k]*(1/k!) multiply is independent of the dep chain and overlaps the
			// FMA, mirroring the raw-loop pattern `tabulated[n+k]*invfact[k] + delta*value`.
			// Also drops the former MSVC C2124 ternary workaround: there is no
			// ternary left to confuse its parser.
			((result = coefficientAccessor(ReversedIndices) * ReciprocalFactorial(ReversedIndices) + delta * result),
			 ...);
			return result;
		}
	}


	template <typename Scalar, std::size_t N, typename CoefficientAccessor>
	SEC_FORCE_INLINE constexpr Scalar UnrolledHornerPowerPolynomial(const CoefficientAccessor& coefficientAccessor,
	                                                                const Scalar delta) noexcept
	{
		return Detail::HornerPolynomial::HornerPowerPolynomial_Fold<Scalar, N>(
		        coefficientAccessor, delta, MakeReversedIndexSequence<N>{});
	}


	template <typename Scalar, std::size_t N, typename CoefficientAccessor>
	SEC_FORCE_INLINE constexpr Scalar HornerPowerPolynomial(const CoefficientAccessor& coefficientAccessor,
	                                                        const Scalar delta) noexcept
	{
		if constexpr (N <= 32)
		{
			return UnrolledHornerPowerPolynomial<Scalar, N>(coefficientAccessor, delta);
		}
		else
		{
			return Detail::HornerPolynomial::HornerPowerPolynomial_Loop<Scalar, N>(coefficientAccessor, delta);
		}
	}


	template <typename Scalar, std::size_t N, typename CoefficientAccessor>
	SEC_FORCE_INLINE constexpr Scalar UnrolledHornerTaylorPolynomial(const CoefficientAccessor& coefficientAccessor,
	                                                                 const Scalar delta) noexcept
	{
		return Detail::HornerPolynomial::HornerTaylorPolynomial_Fold<Scalar, N>(
		        coefficientAccessor, delta, MakeReversedIndexSequence<N>{});
	}


	template <typename Scalar, std::size_t N, typename CoefficientAccessor>
	SEC_FORCE_INLINE constexpr Scalar HornerTaylorPolynomial(const CoefficientAccessor& coefficientAccessor,
	                                                         const Scalar delta) noexcept
	{
		if constexpr (N <= 32)
		{
			return UnrolledHornerTaylorPolynomial<Scalar, N>(coefficientAccessor, delta);
		}
		else
		{
			return Detail::HornerPolynomial::HornerTaylorPolynomial_Loop<Scalar, N>(coefficientAccessor, delta);
		}
	}
}
