// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Andy Brown

#pragma once

#include <SecUtility/Macro/ForceInline.hpp>
#include <SecUtility/Meta/IntegerSequence.hpp>

#include <type_traits>
#include <utility>


namespace SecUtility
{
	template <typename Index, typename Operation>
	constexpr SEC_FORCE_INLINE void StaticForeach(std::integer_sequence<Index>, Operation&&) noexcept
	{
		/* NO-OP */
	}

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winvalid-constexpr"
#endif
	template <typename Index, Index... Indices, typename Operation>
	constexpr SEC_FORCE_INLINE auto StaticForeach(
	        std::integer_sequence<Index, Indices...>,
	        Operation&& operation) noexcept(noexcept((operation.template operator()<Indices>(), ...)))
	        -> std::enable_if_t<(std::is_void_v<decltype(operation.template operator()<Indices>())> && ...)>
	{
		(operation.template operator()<Indices>(), ...);
	}

	template <typename Index, Index... Indices, typename Operation>
	constexpr SEC_FORCE_INLINE auto StaticForeach(std::integer_sequence<Index, Indices...>,
	                                              Operation&& operation) noexcept((noexcept(operation(Indices)) && ...))
	        -> std::enable_if_t<std::is_void_v<decltype(operation(std::declval<Index>()))>>
	{
		(operation(Indices), ...);
	}

	template <auto UpperBound,
	          typename Operation,
	          typename Sequence = MakeIntegerSequence<decltype(UpperBound), UpperBound>>
	constexpr SEC_FORCE_INLINE auto StaticForeachInRange(Operation&& op) noexcept(
	        noexcept(StaticForeach(Sequence{}, std::forward<Operation>(op))))
	        -> decltype(StaticForeach(Sequence{}, std::forward<Operation>(op)))
	{
		StaticForeach(Sequence{}, std::forward<Operation>(op));
	}

	// Bounds determine the index type; the step retains its independent sign and width.
	// MakeIntegerSequence rejects bounds that cannot be represented by their common type.
	template <auto Start,
	          auto End,
	          auto StepSize = 1,
	          typename Operation,
	          typename Sequence =
	                  MakeIntegerSequence<std::common_type_t<decltype(Start), decltype(End)>, Start, End, StepSize>>
	constexpr SEC_FORCE_INLINE auto StaticForeachInRange(Operation&& op) noexcept(
	        noexcept(StaticForeach(Sequence{}, std::forward<Operation>(op))))
	        -> decltype(StaticForeach(Sequence{}, std::forward<Operation>(op)))
	{
		StaticForeach(Sequence{}, std::forward<Operation>(op));
	}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

	template <typename Index, typename Operation>
	[[deprecated("Use StaticForeach instead")]]
	constexpr SEC_FORCE_INLINE void StaticForeachWithRuntimeIndex(std::integer_sequence<Index>, Operation&&) noexcept
	{
		/* NO-OP */
	}

	template <typename Index, Index... Indices, typename Operation>
	[[deprecated("Use StaticForeach instead")]]
	constexpr SEC_FORCE_INLINE void StaticForeachWithRuntimeIndex(std::integer_sequence<Index, Indices...>,
	                                                              Operation&& op) noexcept(noexcept((op(Indices), ...)))
	{
		(op(Indices), ...);
	}

	template <auto UpperBound, typename Operation>
	[[deprecated("Use StaticForeachInRange instead")]]
	constexpr SEC_FORCE_INLINE void StaticForeachInRangeWithRuntimeIndex(Operation&& op) noexcept(
	        noexcept(StaticForeachWithRuntimeIndex(std::make_integer_sequence<decltype(UpperBound), UpperBound>{},
	                                               std::forward<Operation>(op))))
	{
		StaticForeachWithRuntimeIndex(std::make_integer_sequence<decltype(UpperBound), UpperBound>{},
		                              std::forward<Operation>(op));
	}

#if defined(__cpp_generic_lambdas) && __cpp_generic_lambdas >= 201707L
	template <auto LowerBound, auto UpperBound, typename Operation>
	constexpr SEC_FORCE_INLINE void StaticForeachInRangeWithRuntimeIndex(Operation&& op)
	{
		using Index = std::common_type_t<std::decay_t<decltype(LowerBound)>, std::decay_t<decltype(UpperBound)>>;
		static_assert(std::is_integral_v<Index>);
		static_assert(LowerBound <= UpperBound);

		if constexpr (LowerBound == UpperBound)
		{
			(void)op;
		}
		else
		{
			[&]<Index... Indices>(std::integer_sequence<Index, Indices...>)
			{ (op(Indices + LowerBound), ...); }(std::make_integer_sequence<Index, UpperBound - LowerBound>{});
		}
	}
#endif
}