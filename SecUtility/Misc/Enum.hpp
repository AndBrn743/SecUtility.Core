// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <utility>


#if !(defined(__cpp_lib_to_underlying) && __cpp_lib_to_underlying >= 202102L)
namespace std
{
	template <class Enum>
	constexpr std::underlying_type_t<Enum> to_underlying(const Enum e) noexcept
	{
		return static_cast<std::underlying_type_t<Enum>>(e);
	}
}
#endif
