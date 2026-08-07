// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Macro/ForceInline.hpp>

#include <array>
#include <cstddef>


namespace SecUtility::Math
{
	namespace Detail::EstrinPolynomial
	{
		// power: ∑ c_k x^k
		// taylor: ∑ c_k x^k / k!
		//
		// Estrin's scheme rewrites a degree-(N-1) polynomial as a polynomial
		// in y = x^2 whose coefficients are d_k = c_{2k} + c_{2k+1} x, then
		// recurses.  This trades Horner's O(N) serial dependency chain for an
		// O(log N) depth tree of independent pairwise FMAs, exposing ILP at
		// the cost of O(N) total work (same asymptotic count as Horner).

		template <typename Scalar, std::size_t N, typename CoefficientAccessor>
		constexpr Scalar EstrinPowerPolynomial_Loop(const CoefficientAccessor& coefficientAccessor,
		                                            const Scalar delta) noexcept
		{
			static_assert(N > 0);

			std::array<Scalar, N> buf{};
			for (std::size_t i = 0; i < N; ++i)
			{
				buf[i] = coefficientAccessor(i);
			}

			Scalar xs = delta;
			std::size_t m = N;
			while (m > 1)
			{
				const std::size_t half = (m + 1) / 2;
				for (std::size_t k = 0; k < half; ++k)
				{
					const std::size_t lo = 2 * k;
					const std::size_t hi = lo + 1;
					if (hi < m)
					{
						// FMA-friendly: buf[lo] + buf[hi]*xs contracts to
						// fma(buf[hi], xs, buf[lo]) under -ffp-contract=on.
						buf[k] = buf[lo] + buf[hi] * xs;
					}
					else
					{
						buf[k] = buf[lo];
					}
				}
				m = half;
				xs = xs * xs;
			}

			return buf[0];
		}

		template <typename Scalar, std::size_t N, typename CoefficientAccessor>
		SEC_FORCE_INLINE constexpr Scalar EstrinPowerPolynomial_Recursive(
		        const CoefficientAccessor& coefficientAccessor, const Scalar delta) noexcept
		{
			static_assert(N > 0);

			if constexpr (N == 1)
			{
				return coefficientAccessor(0);
			}
			else if constexpr (N == 2)
			{
				// FMA-friendly: c0 + c1*x contracts to fma(c1, x, c0).
				return coefficientAccessor(0) + coefficientAccessor(1) * delta;
			}
			else
			{
				constexpr std::size_t M = (N + 1) / 2;
				auto pairAccessor = [&coefficientAccessor, delta](const std::size_t k) -> Scalar
				{
					// Odd N leaves the top coefficient unpaired; elide the high
					// multiply-add entirely rather than synthesizing a stray *0.
					if constexpr (N % 2 == 1)
					{
						if (k == M - 1)
						{
							return coefficientAccessor(2 * k);
						}
					}
					// FMA-friendly: c[2k] + c[2k+1]*x contracts to
					// fma(c[2k+1], x, c[2k]) under -ffp-contract=on.
					return coefficientAccessor(2 * k) + coefficientAccessor(2 * k + 1) * delta;
				};
				return EstrinPowerPolynomial_Recursive<Scalar, M>(pairAccessor, delta * delta);
			}
		}

		template <typename Scalar, std::size_t N, typename CoefficientAccessor>
		constexpr Scalar EstrinTaylorPolynomial_Loop(const CoefficientAccessor& coefficientAccessor,
		                                             const Scalar delta) noexcept
		{
			static_assert(!std::is_integral_v<Scalar>);
			static_assert(N > 0);

			// Compile-time table of 1/0!, 1/1!, ..., 1/(N-1)!.
			constexpr auto InversedFactorial = []() constexpr
			{
				std::array<Scalar, N> arr{};
				arr[0] = Scalar{1};
				for (std::size_t i = 1; i < N; ++i)
				{
					arr[i] = arr[i - 1] / static_cast<Scalar>(i);
				}
				return arr;
			}();

			std::array<Scalar, N> buf{};
			for (std::size_t i = 0; i < N; ++i)
			{
				// c[k]*(1/k!) is independent of the Estrin dependency tree
				// and overlaps it, mirroring the Horner Taylor pattern.
				buf[i] = coefficientAccessor(i) * InversedFactorial[i];
			}

			Scalar xs = delta;
			std::size_t m = N;
			while (m > 1)
			{
				const std::size_t half = (m + 1) / 2;
				for (std::size_t k = 0; k < half; ++k)
				{
					const std::size_t lo = 2 * k;
					const std::size_t hi = lo + 1;
					if (hi < m)
					{
						buf[k] = buf[lo] + buf[hi] * xs;
					}
					else
					{
						buf[k] = buf[lo];
					}
				}
				m = half;
				xs = xs * xs;
			}

			return buf[0];
		}

		template <typename Scalar, std::size_t N, typename CoefficientAccessor>
		SEC_FORCE_INLINE constexpr Scalar EstrinTaylorPolynomial_Recursive(
		        const CoefficientAccessor& coefficientAccessor, const Scalar delta) noexcept
		{
			static_assert(!std::is_integral_v<Scalar>);
			static_assert(N > 0);

			// Compile-time table of 1/0!, 1/1!, ..., 1/(N-1)!.
			constexpr auto InversedFactorial = []() constexpr
			{
				std::array<Scalar, N> arr{};
				arr[0] = Scalar{1};
				for (std::size_t i = 1; i < N; ++i)
				{
					arr[i] = arr[i - 1] / static_cast<Scalar>(i);
				}
				return arr;
			}();

			// Scale each c[k] by 1/k! on the fly and reuse the power Estrin
			// tree.  InversedFactorial[k] folds to an immediate once the
			// recursive template unrolls, since each leaf call site has a
			// constant k after inlining.
			auto scaledAccessor = [&coefficientAccessor, &InversedFactorial](std::size_t k) -> Scalar
			{ return coefficientAccessor(k) * InversedFactorial[k]; };

			return EstrinPowerPolynomial_Recursive<Scalar, N>(scaledAccessor, delta);
		}
	}


	template <typename Scalar, std::size_t N, typename CoefficientAccessor>
	SEC_FORCE_INLINE constexpr Scalar UnrolledEstrinPowerPolynomial(const CoefficientAccessor& coefficientAccessor,
	                                                                const Scalar delta) noexcept
	{
		return Detail::EstrinPolynomial::EstrinPowerPolynomial_Recursive<Scalar, N>(coefficientAccessor, delta);
	}


	template <typename Scalar, std::size_t N, typename CoefficientAccessor>
	SEC_FORCE_INLINE constexpr Scalar EstrinPowerPolynomial(const CoefficientAccessor& coefficientAccessor,
	                                                        const Scalar delta) noexcept
	{
		if constexpr (N <= 32)
		{
			return UnrolledEstrinPowerPolynomial<Scalar, N>(coefficientAccessor, delta);
		}
		else
		{
			return Detail::EstrinPolynomial::EstrinPowerPolynomial_Loop<Scalar, N>(coefficientAccessor, delta);
		}
	}


	template <typename Scalar, std::size_t N, typename CoefficientAccessor>
	SEC_FORCE_INLINE constexpr Scalar UnrolledEstrinTaylorPolynomial(const CoefficientAccessor& coefficientAccessor,
	                                                                 const Scalar delta) noexcept
	{
		return Detail::EstrinPolynomial::EstrinTaylorPolynomial_Recursive<Scalar, N>(coefficientAccessor, delta);
	}


	template <typename Scalar, std::size_t N, typename CoefficientAccessor>
	SEC_FORCE_INLINE constexpr Scalar EstrinTaylorPolynomial(const CoefficientAccessor& coefficientAccessor,
	                                                         const Scalar delta) noexcept
	{
		if constexpr (N <= 32)
		{
			return UnrolledEstrinTaylorPolynomial<Scalar, N>(coefficientAccessor, delta);
		}
		else
		{
			return Detail::EstrinPolynomial::EstrinTaylorPolynomial_Loop<Scalar, N>(coefficientAccessor, delta);
		}
	}
}
