// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once


namespace SecUtility
{
	/// polyfill of the C++23 <c>std::unreachable()</c>.
	/// see n4950/[utility.unreachable] and https://en.cppreference.com/cpp/utility/unreachable
	[[noreturn]] inline void Unreachable()
	{
#if defined(_MSC_VER) && !defined(__clang__)
		__assume(false);
#else
		__builtin_unreachable();
#endif
	}
}
