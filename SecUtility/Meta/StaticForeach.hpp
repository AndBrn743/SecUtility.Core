// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Andy Brown

#pragma once

#include <SecUtility/Macro/ForceInline.hpp>

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
#pragma GCC diagnostic ignored "-Winvalid-constexpr"
#endif
	template <typename Index, Index... Indices, typename Operation>
	constexpr SEC_FORCE_INLINE void StaticForeach(
	        std::integer_sequence<Index, Indices...>,
	        Operation&& operation) noexcept(noexcept((operation.template operator()<Indices>(), ...)))
	{
		(operation.template operator()<Indices>(), ...);
	}

	template <auto UpperBound, typename Operation>
	constexpr SEC_FORCE_INLINE void StaticForeachInRange(Operation&& op)
	{
		using Index = std::decay_t<decltype(UpperBound)>;
		static_assert(std::is_integral_v<Index>);
		static_assert(UpperBound >= 0);

		if constexpr (UpperBound == 0)
		{
			(void)op;
		}
		else
		{
			StaticForeach(std::make_integer_sequence<Index, UpperBound>{}, std::forward<Operation>(op));
		}
	}

#if defined(__cpp_generic_lambdas) && __cpp_generic_lambdas >= 201707L
	template <auto LowerBound, auto UpperBound, typename Operation>
	constexpr SEC_FORCE_INLINE void StaticForeachInRange(Operation&& op)
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
			{
				(op.template operator()<Indices + LowerBound>(), ...);
			}(std::make_integer_sequence<Index, UpperBound - LowerBound>{});
		}
	}

	template <auto Start, auto End, auto StepSize, typename Operation>
	constexpr SEC_FORCE_INLINE void StaticForeachInRange(Operation&& op)
	{
		using Index = std::common_type_t<std::decay_t<decltype(Start)>,
		                                 std::decay_t<decltype(End)>,
		                                 std::decay_t<decltype(StepSize)>>;
		static_assert(std::is_integral_v<Index>);
		static_assert(StepSize != 0);
		static_assert((StepSize > 0 && Start <= End) || (StepSize < 0 && Start >= End));

		constexpr Index StepCount = (End - Start) / StepSize + ((End - Start) % StepSize != 0 ? 1 : 0);

		if constexpr (StepCount == 0)
		{
			(void)op;
		}
		else
		{
			[&]<typename Index, Index... Indices>(std::integer_sequence<Index, Indices...>)
			{
				(op.template operator()<Start + StepSize * Indices>(), ...);
			}(std::make_integer_sequence<Index, StepCount>{});
		}
	}
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

	template <typename Index, typename Operation>
	constexpr SEC_FORCE_INLINE void StaticForeachWithRuntimeIndex(std::integer_sequence<Index>, Operation&&) noexcept
	{
		/* NO-OP */
	}

	template <typename Index, Index... Indices, typename Operation>
	constexpr SEC_FORCE_INLINE void StaticForeachWithRuntimeIndex(std::integer_sequence<Index, Indices...>,
	                                                              Operation&& op) noexcept(noexcept((op(Indices), ...)))
	{
		(op(Indices), ...);
	}

	template <auto UpperBound, typename Operation>
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