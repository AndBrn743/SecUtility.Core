// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Meta/OverloadSet.hpp>
#include <algorithm>
#include <cctype>
#include <string>
#include <utility>


namespace SecUtility
{
	namespace Detail::CaseConversion
	{
		inline constexpr auto ToLower = [](const unsigned char c)
		{ return static_cast<unsigned char>(std::tolower(c)); };

		inline constexpr auto ToUpper = [](const unsigned char c)
		{ return static_cast<unsigned char>(std::toupper(c)); };

		inline constexpr auto AsciiToLower = [](const unsigned char c) noexcept
		{ return c >= 'A' && c <= 'Z' ? static_cast<unsigned char>(c + ('a' - 'A')) : c; };

		inline constexpr auto AsciiToUpper = [](const unsigned char c) noexcept
		{ return c >= 'a' && c <= 'z' ? static_cast<unsigned char>(c + ('A' - 'a')) : c; };
	}

#define SEC_POPULATE_CASE_CONVERSION(OP)                                                                               \
	inline constexpr auto OP =                                                                                         \
	        OverloadSet{Detail::CaseConversion::OP,                                                                    \
	                    [](std::string s)                                                                              \
	                    {                                                                                              \
		                    std::transform(s.begin(), s.end(), s.begin(), Detail::CaseConversion::OP);                 \
		                    return s;                                                                                  \
	                    },                                                                                             \
	                    [](std::string& ref_s, std::in_place_t)                                                        \
	                    { std::transform(ref_s.begin(), ref_s.end(), ref_s.begin(), Detail::CaseConversion::OP); }};

	SEC_POPULATE_CASE_CONVERSION(ToLower)
	SEC_POPULATE_CASE_CONVERSION(ToUpper)
	SEC_POPULATE_CASE_CONVERSION(AsciiToLower)
	SEC_POPULATE_CASE_CONVERSION(AsciiToUpper)

#undef SEC_POPULATE_CASE_CONVERSION
}
