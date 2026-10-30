// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace SecUtility
{
	template <std::size_t N, typename Scalar>
	constexpr std::array<Scalar, N> MakeGeometricSequenceArray(const Scalar initial, const Scalar ratio) noexcept(
	        noexcept(std::array<Scalar, N>{initial}) && std::is_nothrow_move_constructible_v<std::array<Scalar, N>>
	        && (N == 1 || noexcept(std::declval<Scalar&>() = ratio * std::declval<Scalar&>())))
	{
		static_assert(N > 0);

		std::array<Scalar, N> result{initial};

		for (std::size_t i = 1; i < N; i++)  // `i` start with 1
		{
			result[i] = ratio * result[i - 1];
		}

		return result;
	}

	template <std::size_t N, typename Scalar>
	constexpr std::array<Scalar, N> MakeArithmeticSequenceArray(const Scalar initial, const Scalar delta) noexcept(
	        noexcept(std::array<Scalar, N>{initial}) && std::is_nothrow_move_constructible_v<std::array<Scalar, N>>
	        && (N == 1 || noexcept(std::declval<Scalar&>() = std::declval<Scalar&>() + delta)))
	{
		static_assert(N > 0);

		std::array<Scalar, N> result{initial};

		for (std::size_t i = 1; i < N; i++)  // `i` start with 1
		{
			result[i] = result[i - 1] + delta;
		}

		return result;
	}
}
