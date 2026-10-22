// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if __cplusplus >= 202002L

#include <cstddef>
#include <utility>

namespace SecUtility
{
	namespace Detail::NttpArray
	{
		template <template <auto...> class Template, auto Array, std::size_t... Is>
		// ReSharper disable once CppFunctionIsNotImplemented
		/* NOT IMPLEMENTED */ auto ApplyNttpArray(std::index_sequence<Is...>) -> Template<Array[Is]...>;

		template <auto& Array, typename F, std::size_t... Is>
		// ReSharper disable once CppFunctionIsNotImplemented
		/* NOT IMPLEMENTED */ constexpr auto ApplyNttpArrayToGenerator(F f, std::index_sequence<Is...>)
		        -> decltype(f.template operator()<Array[Is]...>());
	}

	template <template <auto...> class Template, auto Array>
	using ApplyNttpArrayTo =
	        decltype(Detail::NttpArray::ApplyNttpArray<Template, Array>(std::make_index_sequence<Array.size()>{}));

	template <typename F, auto Array>
	using ApplyNttpArrayToGenerator = decltype(Detail::NttpArray::ApplyNttpArrayToGenerator<Array>(
	        F{}, std::make_index_sequence<Array.size()>{}));
}

#endif
