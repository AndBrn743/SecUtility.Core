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

		template <typename T, typename U, typename... Vs>
		struct concat<T, U, Vs...>
		{
			using type = concat<typename concat<T, U>::type, Vs...>;
		};

		template <typename... Ts>
		struct concat<SecUtility::TypeTuple<Ts...>>
		{
			using type = SecUtility::TypeTuple<Ts...>;
		};

		template <typename... Ts, typename... Us>
		struct concat<SecUtility::TypeTuple<Ts...>, SecUtility::TypeTuple<Us...>>
		{
			using type = SecUtility::TypeTuple<Ts..., Us...>;
		};
	}


	template <typename... Ts>
	using ConcatenatedTypeTuple = typename Detail::TypeTuple::concat<Ts...>::type;
}
