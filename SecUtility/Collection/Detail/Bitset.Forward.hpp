// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <cstddef>


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
	}
}
