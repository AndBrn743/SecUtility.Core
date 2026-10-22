// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <utility>


namespace SecUtility
{
	// SecUtility implementation of std::identity from C++20. See https://wg21.link/std20 [func.identity]
	struct Identity
	{
		using is_transparent = void;

		template <typename T>
		constexpr T&& operator()(T&& t) const noexcept
		{
			return std::forward<T>(t);
		}
	};
}
