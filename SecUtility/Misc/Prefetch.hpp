// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif

#include <SecUtility/Macro/ConstevalIf.hpp>


namespace SecUtility
{

#if defined(SEC_IF_NOT_CONSTEVAL)
	constexpr
#else
	inline
#endif
			void Prefetch(const void* ptr) noexcept
	{
#if defined(SEC_IF_NOT_CONSTEVAL)
		SEC_IF_NOT_CONSTEVAL
#else
		if constexpr (true)
#endif
		{
#if defined(__GNUC__) || defined(__clang__)
			__builtin_prefetch(ptr);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
			_mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#else
			(void)ptr;
#endif
		}
		else
		{
			(void)ptr;
		}
	}


#if defined(SEC_IF_NOT_CONSTEVAL)
	constexpr
#else
	inline
#endif
			void PrefetchStream(const void* ptr) noexcept
	{
#if defined(SEC_IF_NOT_CONSTEVAL)
		SEC_IF_NOT_CONSTEVAL
#else
		if constexpr (true)
#endif
		{
#if defined(__GNUC__) || defined(__clang__)
			__builtin_prefetch(ptr, 0, 0);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
			_mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_NTA);
#else
			(void)ptr;
#endif
		}
		else
		{
			(void)ptr;
		}
	}
}  // namespace SecUtility
