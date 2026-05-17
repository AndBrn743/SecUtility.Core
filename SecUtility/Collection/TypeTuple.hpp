// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

namespace SecUtility
{
	template <typename...>
	/* INCOMPLETE */ struct TypeTuple;

	template <typename T, typename U>
	using TypePair = TypeTuple<T, U>;
}
