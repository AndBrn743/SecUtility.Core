// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <cstddef>
#include <type_traits>


namespace SecUtility
{
	template <typename>
	struct Traits;
}


#if !(defined(__cpp_lib_bounded_array_traits) && __cpp_lib_bounded_array_traits >= 201902L)
namespace std
{
	template <typename>
	struct is_bounded_array : std::false_type
	{
	};

	template <typename T, std::size_t S>
	struct is_bounded_array<T[S]> : std::true_type
	{
	};

	template <typename T>
    inline constexpr bool is_bounded_array_v = is_bounded_array<T>::value;
}
#endif

#if !(defined(__cpp_lib_unbounded_array_traits) && __cpp_lib_unbounded_array_traits >= 201902L)
namespace std
{
	template <typename>
	struct is_unbounded_array : std::false_type
	{
	};

	template <typename T>
	struct is_unbounded_array<T[]> : std::true_type
	{
	};

	template <typename T>
    inline constexpr bool is_unbounded_array_v = is_unbounded_array<T>::value;
}
#endif
