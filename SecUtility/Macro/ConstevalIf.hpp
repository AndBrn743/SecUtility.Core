// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once


// clang-format off

#if defined(__cpp_if_consteval) && __cpp_if_consteval >= 202106L

	#define SEC_IF_CONSTEVAL if consteval
	#define SEC_IF_NOT_CONSTEVAL if !consteval

#elif defined(__cpp_lib_is_constant_evaluated) && __cpp_lib_is_constant_evaluated >= 201811L

	#include <type_traits>
	#define SEC_IF_CONSTEVAL if /*constexpr*/ (std::is_constant_evaluated())
	#define SEC_IF_NOT_CONSTEVAL if /*constexpr*/ (!std::is_constant_evaluated())

#elif (defined(__INTEL_COMPILER) && __INTEL_COMPILER >= 20210000) || (defined(__clang__) && __clang_major__ >= 9)      \
        || (defined(__GNUC__) && !defined(__clang__) && (__GNUC__ > 9 || (__GNUC__ == 9 && __GNUC_MINOR__ >= 1)))      \
        || (defined(_MSC_VER) && _MSC_VER >= 1925)

	#define SEC_IF_CONSTEVAL if /*constexpr*/ (__builtin_is_constant_evaluated())
	#define SEC_IF_NOT_CONSTEVAL if /*constexpr*/ (!__builtin_is_constant_evaluated())

#elif defined(__has_builtin) && __has_builtin(__builtin_is_constant_evaluated)

	#define SEC_IF_CONSTEVAL if /*constexpr*/ (__builtin_is_constant_evaluated())
	#define SEC_IF_NOT_CONSTEVAL if /*constexpr*/ (!__builtin_is_constant_evaluated())

#endif

// clang-format on
