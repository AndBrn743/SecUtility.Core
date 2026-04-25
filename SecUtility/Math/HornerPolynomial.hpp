// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Macro/ForceInline.hpp>
#include <SecUtility/Meta/IntegerSequence.hpp>


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
			((result = (result + coefficientAccessor(ReversedIndices)) * (ReversedIndices == 0 ? 1 : delta)), ...);
			return result;
		}

		template <typename Scalar, std::size_t N, typename CoefficientAccessor>
		constexpr Scalar HornerTaylorPolynomial_Loop(const CoefficientAccessor& coefficientAccessor,
		                                             const Scalar delta) noexcept
		{
			static_assert(N > 0);
			Scalar result = 0;

			for (std::size_t reversedIndices = N; reversedIndices-- > 1;)
			{
				result += coefficientAccessor(reversedIndices);
				result *= delta / reversedIndices;
			}

			result += coefficientAccessor(0);

			return result;
		}

		template <typename Scalar, std::size_t N, typename CoefficientAccessor, std::size_t... ReversedIndices>
		SEC_FORCE_INLINE constexpr Scalar HornerTaylorPolynomial_Fold(const CoefficientAccessor& coefficientAccessor,
		                                                              const Scalar delta,
		                                                              std::index_sequence<ReversedIndices...>) noexcept
		{
			static_assert(N > 0);
			static_assert(sizeof...(ReversedIndices) == N);
			static_assert(((ReversedIndices < N) && ...));
			Scalar result = 0;
			((result = (result + coefficientAccessor(ReversedIndices))
			           * (ReversedIndices == 0 ? 1 : delta / ReversedIndices)),
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
