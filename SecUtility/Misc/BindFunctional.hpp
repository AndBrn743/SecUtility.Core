//
// Created by Andy on 5/9/2026.
//

#pragma once

#include <SecUtility/Meta/TypeTrait.hpp>
#include <functional>

namespace SecUtility::Polyfill::Detail
{
	template <class T, class U>
	struct copy_const : std::conditional<std::is_const_v<T>, U const, U>
	{
	};

	template <class T, class U, class X = typename copy_const<std::remove_reference_t<T>, U>::type>
	struct copy_value_category : std::conditional<std::is_lvalue_reference_v<T&&>, X&, X&&>
	{
	};

	template <class T, class U>
	struct type_forward_like : copy_value_category<T, std::remove_reference_t<U>>
	{
	};

	template <class T, class U>
	using type_forward_like_t = typename type_forward_like<T, U>::type;
}

namespace SecUtility
{
	template <auto ConstFn, class... Args>
	constexpr auto bind_front(Args&&... args)
	{
		using F = decltype(ConstFn);

		if constexpr (std::is_pointer_v<F> or std::is_member_pointer_v<F>)
		{
			static_assert(ConstFn != nullptr);
		}

		return [... bound_args(std::forward<Args>(args))]<class Self, class... T>(
		               this Self&&,
		               T&&... call_args) noexcept(std::is_nothrow_invocable_v<F,
		                                                                      detail::type_forward_like_t<
		                                                                              Self,
		                                                                              std::decay_t<Args>>...,
		                                                                      T...>)
		               -> std::invoke_result_t<F, detail::type_forward_like_t<Self, std::decay_t<Args>>..., T...>
		{ return std::invoke(ConstFn, std::forward_like<Self>(bound_args)..., std::forward<T>(call_args)...); };
	}
}