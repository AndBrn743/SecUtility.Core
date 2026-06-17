// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

namespace SecUtility
{
	template <typename...>
	/* INCOMPLETE */ struct TypeTuple;

	template <typename T, typename U>
	using TypePair = TypeTuple<T, U>;

	namespace Detail::TypeTuple
	{
		template <typename...>
		struct concat;

		template <typename... Ts>
		struct concat<SecUtility::TypeTuple<Ts...>>
		{
			using type = SecUtility::TypeTuple<Ts...>;
		};

		template <typename... Ts, typename... Us, typename... Rest>
		struct concat<SecUtility::TypeTuple<Ts...>, SecUtility::TypeTuple<Us...>, Rest...>
		{
			using type = typename concat<SecUtility::TypeTuple<Ts..., Us...>, Rest...>::type;
		};
	}

	template <typename... Ts>
	using ConcatenatedTypeTuple = typename Detail::TypeTuple::concat<Ts...>::type;
}
