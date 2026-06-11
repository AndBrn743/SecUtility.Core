// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Meta/TypeTrait.hpp>


namespace SecUtility
{
	template <typename>
	struct is_bitmask : std::false_type
	{
		/* NO CODE */
	};

	template <typename E>
	constexpr bool is_bitmask_v = is_bitmask<E>::value;
}

template <typename E>
constexpr std::enable_if_t<SecUtility::is_bitmask_v<E>, E> operator|(const E lhs, const E rhs) noexcept
{
	using U = std::underlying_type_t<E>;
	return static_cast<E>(static_cast<U>(lhs) | static_cast<U>(rhs));
}

template <typename E>
constexpr std::enable_if_t<SecUtility::is_bitmask_v<E>, E> operator&(const E lhs, const E rhs) noexcept
{
	using U = std::underlying_type_t<E>;
	return static_cast<E>(static_cast<U>(lhs) & static_cast<U>(rhs));
}

template <typename E>
constexpr std::enable_if_t<SecUtility::is_bitmask_v<E>, E> operator^(E lhs, const E rhs) noexcept
{
	using U = std::underlying_type_t<E>;
	return static_cast<E>(static_cast<U>(lhs) ^ static_cast<U>(rhs));
}

template <typename E>
constexpr std::enable_if_t<SecUtility::is_bitmask_v<E>, E> operator~(const E v) noexcept
{
	using U = std::underlying_type_t<E>;
	return static_cast<E>(~static_cast<U>(v));
}

template <typename E>
constexpr std::enable_if_t<SecUtility::is_bitmask_v<E>, E&> operator|=(E& lhs, const E rhs) noexcept
{
	return lhs = lhs | rhs;
}

template <typename E>
constexpr std::enable_if_t<SecUtility::is_bitmask_v<E>, E&> operator&=(E& lhs, const E rhs) noexcept
{
	return lhs = lhs & rhs;
}

template <typename E>
constexpr std::enable_if_t<SecUtility::is_bitmask_v<E>, E&> operator^=(E& lhs, const E rhs) noexcept
{
	return lhs = lhs ^ rhs;
}
