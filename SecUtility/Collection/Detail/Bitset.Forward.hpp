// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if !defined(SEC_BITSET_DETAIL)
#error "Please do not directly include internal headers"
#endif

#include <cstddef>
#include <SecUtility/Meta/TypeTrait.hpp>


namespace SecUtility
{
	template <typename Derived>
	class BitsetBase;

	template <std::size_t N>
	class Bitset;

	class DynamicBitset;

	namespace Detail::Bitset
	{
		template <typename Nested>
		class BitsetSegmentExpr;

		template <typename Nested>
		class BitsetNotExpr;

		template <typename Op, typename Lhs, typename Rhs>
		class BitsetBinaryExpr;

		template <typename>
		struct is_fixed_size_bitset : std::false_type
		{
			/* NO CODE */
		};

		template <std::size_t N>
		struct is_fixed_size_bitset<SecUtility::Bitset<N>> : std::true_type
		{
			/* NO CODE */
		};
	}
}
