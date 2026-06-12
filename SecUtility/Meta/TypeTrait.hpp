// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Collection/TypeTuple.hpp>
#include <type_traits>
#include <array>
#include <cstddef>


namespace SecUtility
{
	template <typename>
	struct Traits;

	template <typename>
	struct FunctionTraits;

	template <typename R, typename... Args>
	struct FunctionTraits<R(Args...)>
	{
		static constexpr std::size_t Arity = sizeof...(Args);
		using ReturnType = R;
		using ArgTypeTuple = TypeTuple<Args...>;
	};

	template <typename R, typename... Args>
	struct FunctionTraits<R (*)(Args...)> : FunctionTraits<R(Args...)>
	{
		/* NO CODE */
	};

	template <typename C, typename R, typename... Args>
	struct FunctionTraits<R (C::*)(Args...) const> : FunctionTraits<R(Args...)>
	{
		/* NO CODE */
	};

	template <typename C, typename R, typename... Args>
	struct FunctionTraits<R (C::*)(Args...)> : FunctionTraits<R(Args...)>
	{
		/* NO CODE */
	};

	template <typename Functor>
	struct FunctionTraits : FunctionTraits<decltype(&std::remove_reference_t<Functor>::operator())>
	{
		/* NO CODE */
	};
}


#if !(defined(__cpp_lib_bounded_array_traits) && __cpp_lib_bounded_array_traits >= 201902L)
namespace std
{
	template <typename>
	struct is_bounded_array : std::false_type
	{
	};

	template <typename T, std::size_t S>
	struct is_bounded_array<T[S]> : std::true_type
	{
	};

	template <typename T>
	inline constexpr bool is_bounded_array_v = is_bounded_array<T>::value;

	template <typename>
	struct is_unbounded_array : std::false_type
	{
	};

	template <typename T>
	struct is_unbounded_array<T[]> : std::true_type
	{
	};

	template <typename T>
	inline constexpr bool is_unbounded_array_v = is_unbounded_array<T>::value;
}
#endif

#if !(defined(__cpp_lib_type_identity) && __cpp_lib_type_identity >= 201806L)
namespace std
{
	template <typename T>
	struct type_identity
	{
		using type = T;
	};

	template <typename T>
	using type_identity_t = typename type_identity<T>::type;
}
#endif

#if !(defined(__cpp_lib_remove_cvref) && __cpp_lib_remove_cvref >= 201711L)
namespace std
{
	template <class T>
	struct remove_cvref
	{
		using type = std::remove_cv_t<std::remove_reference_t<T>>;
	};

	template <class T>
	using remove_cvref_t = typename remove_cvref<T>::type;
}
#endif
