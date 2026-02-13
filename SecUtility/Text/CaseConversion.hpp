// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <cctype>


namespace SecUtility
{
	inline constexpr auto ToLower = [](const unsigned char c) noexcept { return std::tolower(c); };
	inline constexpr auto ToUpper = [](const unsigned char c) noexcept { return std::toupper(c); };

	inline constexpr auto AsciiToLower = [](const unsigned char c) noexcept
	{ return (c >= 'A' && c <= 'Z') ? (c + ('a' - 'A')) : c; };
	inline constexpr auto AsciiToUpper = [](const unsigned char c) noexcept
	{ return (c >= 'a' && c <= 'z') ? (c + ('A' - 'a')) : c; };
}
