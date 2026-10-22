// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 Andy Brown

#pragma once

#include <SecUtility/Macro/ForceInline.hpp>
#include <SecUtility/Raw/Int.hpp>
#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <range/v3/view/slice.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif


namespace SecUtility::Math
{
	// Two overloads: explicit element type via CalculatePascalRow<T>(n, out), or T deduced from n.
	// The empty pack between Scalar and Int permits explicit Scalar specification while still deducing Int from n.
	/// <summary>
	/// Writes row n of Pascal's Triangle (n+1 values) to out via the multiplicative recurrence C(n,k+1) = C(n,k) *
	/// (n-k) / (k+1).
	/// </summary>
	/// <remarks>
	/// Element type is the explicit template argument Scalar when supplied, otherwise deduced from n.
	/// Requires n &gt;= 0 (asserted) and Int to be an integral type (static_asserted).
	/// Avoids forming n! first; the recurrence stays inside Int64 through row 61.
	/// </remarks>
	template <typename Scalar, typename..., typename Int, typename OutputIterator>
	constexpr std::enable_if_t<!std::is_same_v<Scalar, void>> CalculatePascalRow(const Int n, OutputIterator out)
	{
		static_assert(std::is_integral_v<Int>);
		assert(n >= 0);

		UInt64 value = 1;
		*out = value;
		++out;
		for (UInt64 k = 0; k < static_cast<UInt64>(n); ++k)
		{
			value = value * (n - k) / (k + 1);
			*out = static_cast<Scalar>(value);
			++out;
		}
	}

	template <typename X = void, typename..., typename Int, typename OutputIterator>
	constexpr std::enable_if_t<std::is_same_v<X, void>> CalculatePascalRow(const Int n, OutputIterator out)
	{
		CalculatePascalRow<Int>(n, out);
	}

	namespace Detail::PascalTriangle
	{
		// Largest n for which the multiplicative recurrence stays inside Int64.
		// At row 62 the intermediate C(62,30) * 32 overflows even though the row's final entries fit.
		inline constexpr std::size_t MaxRow = 61;

		// Flat storage: row i occupies indices [i*(i+1)/2, (i+1)*(i+2)/2). Total = 62*63/2 = 1953 entries.
		inline constexpr auto Rows = []() constexpr
		{
			std::array<Int64, (MaxRow + 1) * (MaxRow + 2) / 2> data = {};

			for (Int64 n = 0; n <= static_cast<Int64>(MaxRow); ++n)
			{
				const std::size_t offset = static_cast<std::size_t>(n * (n + 1) / 2);
				CalculatePascalRow<Int64>(n, &data[offset]);
			}

			return data;
		}();
	}

	/// <summary>
	/// Returns row n of Pascal's Triangle from a compile-time table as a (pointer, length) pair.
	/// </summary>
	/// <remarks>
	/// n must be in [0, 61]; the table is built with the multiplicative recurrence which overflows at row 62.
	/// </remarks>
	constexpr SEC_FORCE_INLINE auto PascalRow(const Int64 n) noexcept
	{
		assert(n >= 0 && n <= static_cast<Int64>(Detail::PascalTriangle::MaxRow));
		const auto offset = static_cast<std::size_t>(n * (n + 1) / 2);
		return Detail::PascalTriangle::Rows | ranges::views::slice(offset, offset + static_cast<std::size_t>(n + 1));
	}
}
