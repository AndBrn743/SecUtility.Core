// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 Andy Brown

// Regin indexing for efficient triangular matrix storage. (The name "regin"
// is result from a mistake from the past, I have no plan in fixing it however.)
//
// The Regin functions map 2D symmetric matrix indices to 1D compact storage,
// storing only the upper (or lower) triangular portion. This reduces storage
// by ~50% for symmetric/Hermitian matrices.
//
// For an N * N matrix, element (i, j) with i >= j maps to index: i * (i + 1) / 2 + j (0-based)
// - (0,0) -> 0
// - (1,0) -> 1, (1,1) -> 2
// - (2,0) -> 3, (2,1) -> 4, (2,2) -> 5

#pragma once

#include <SecUtility/Macro/ForceInline.hpp>
#include <SecUtility/Math/Core.hpp>
#include <cassert>

namespace SecUtility::Math
{
	namespace Regin::Detail
	{
		template <typename Int>
		constexpr SEC_FORCE_INLINE Int unsafe_Regin0(const Int i, const Int j) noexcept
		{
			static_assert(std::is_integral_v<std::decay_t<Int>>);
			assert(i >= j);
			assert(j >= 0);
			return (i * (i + 1)) / 2 + j;
		}
	}

	/// Returns the canonical index of natural indices in <b>Natural Number</b> domain
	template <typename Int>
	constexpr SEC_FORCE_INLINE Int Regin0(const Int i, const Int j) noexcept
	{
		return i >= j ? Regin::Detail::unsafe_Regin0(i, j) : Regin::Detail::unsafe_Regin0(j, i);
	}


	/// Returns the canonical index of natural indices in <b>Positive Integer</b> domain
	template <typename Int>
	constexpr SEC_FORCE_INLINE Int Regin1(const Int i, const Int j) noexcept
	{
		return i >= j ? Regin::Detail::unsafe_Regin0(i - 1, j - 1) + 1 : Regin::Detail::unsafe_Regin0(j - 1, i - 1) + 1;
	}


	/// Returns the canonical index of natural indices in <b>Natural Number</b> domain
	template <typename Int>
	constexpr SEC_FORCE_INLINE Int Regin0(const Int i, const Int j, const Int k, const Int l) noexcept
	{
		return Regin0(Regin0(i, j), Regin0(k, l));
	}


	/// Returns the canonical index of natural indices in <b>Positive Integer</b> domain
	template <typename Int>
	constexpr SEC_FORCE_INLINE Int Regin1(const Int i, const Int j, const Int k, const Int l) noexcept
	{
		return Regin1(Regin1(i, j), Regin1(k, l));
	}


	/// Returns the canonical index of natural indices in <b>Natural Number</b> domain
	/// \tparam N Operation Power
	///         <p><code>Reg0\<2\>(i) = Regin0(i, i)</code>
	///         <p><code>Reg0\<4\>(i) = Regin0(i, i, i, i)</code>
	template <std::size_t N, typename Int>
	constexpr SEC_FORCE_INLINE Int Regin0(const Int i) noexcept
	{
		static_assert(N >= 2);
		static_assert(IsPowerOfTwo(N), "Template parameter `N` must be a integer power of two");

		if constexpr (N == 2)
		{
			return Regin::Detail::unsafe_Regin0(i, i);
		}
		else
		{
			const auto temp = Regin::Detail::unsafe_Regin0(i, i);
			return Regin0<N / 2, Int>(temp);
		}
	}

	/// Returns the canonical index of natural indices in <b>Positive Integer</b> domain
	/// \tparam N Operation Power
	///         <p><code>Reg1\<2\>(i) = Regin1(i, i)</code>
	///         <p><code>Reg1\<4\>(i) = Regin1(i, i, i, i)</code>
	template <std::size_t N, typename Int>
	constexpr SEC_FORCE_INLINE Int Regin1(const Int i) noexcept
	{
		return Regin0<N, Int>(i - 1) + 1;
	}
}
