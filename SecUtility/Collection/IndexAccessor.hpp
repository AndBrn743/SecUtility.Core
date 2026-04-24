// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Macro/ForceInline.hpp>
#include <functional>
#include <utility>


namespace SecUtility
{
	/// <summary>
	/// Returns an accessor which expose input <c>indexable</c>'s subscript operator as call operator
	/// </summary>
	/// <remarks>
	/// The input <c>indexable</c> will be taken <b>by value</b>. For lightweight types (e.g. pointers, iterators),
	/// this is typically optimal. To avoid copying or to enforce reference semantics, please wrap the indexable object
	/// with <c>std::ref</c>.
	/// </remarks>
	template <typename Indexable>
	SEC_FORCE_INLINE constexpr auto MakeIndexAccessor(Indexable indexable) noexcept
	{
		return [indexable = std::move(indexable)](const auto index) -> decltype(auto) { return indexable[index]; };
	}


	/// <summary>
	/// Returns an accessor which expose input <c>indexable</c>'s subscript operator as call operator
	/// </summary>
	/// <remarks>
	/// This overload will take input <c>indexable</c> <b>by reference</b>.
	/// </remarks>
	template <typename Indexable>
	SEC_FORCE_INLINE constexpr auto MakeIndexAccessor(std::reference_wrapper<Indexable> indexable) noexcept
	{
		return [&indexable = indexable.get()](const auto index) -> decltype(auto) { return indexable[index]; };
	}
}
