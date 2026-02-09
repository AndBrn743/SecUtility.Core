// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <cstddef>


namespace SecUtility
{
	template <typename T>
	using RawUnboundedArray = T[];

	template <typename T, std::size_t S>
	using RawBoundedArray = T[S];
}
