// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <utility>


namespace SecUtility
{
	namespace Detail::IntegerSequence
	{
		template <typename Sequence>
		struct MakeReversedIntegerSequenceHelper;

		template <typename Int, auto... Is>
		struct MakeReversedIntegerSequenceHelper<std::integer_sequence<Int, Is...>>
		{
			using Type = std::integer_sequence<Int, static_cast<Int>(sizeof...(Is)) - Int{1} - Is...>;
		};
	}

	template <typename T, std::size_t N>
	using MakeIntegerSequence = std::make_integer_sequence<T, N>;

	template <std::size_t N>
	using MakeIndexSequence = std::make_index_sequence<N>;

	template <typename T, std::size_t N>
	using MakeReversedIntegerSequence =
	        typename Detail::IntegerSequence::MakeReversedIntegerSequenceHelper<std::make_integer_sequence<T, N>>::Type;

	template <std::size_t N>
	using MakeReversedIndexSequence = MakeReversedIntegerSequence<std::size_t, N>;
}
