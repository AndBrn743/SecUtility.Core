// SPDX-License-Identifier: MIT
// Copyright (c) 2022-2026 Andy Brown

#pragma once

#include <cstddef>
#include <string_view>

#include <SecUtility/Text/CaseConversion.hpp>


namespace SecUtility
{
	template <typename CharLowerer>
	constexpr int CaseInsensitiveComparison(const std::string_view lhs,
	                                        const std::string_view rhs,
	                                        const CharLowerer lowerCaseOf) noexcept
	{
		const std::size_t minSize = lhs.size() < rhs.size() ? lhs.size() : rhs.size();

		for (std::size_t i = 0; i < minSize; i++)
		{
			const auto la = lowerCaseOf(static_cast<unsigned char>(lhs[i]));
			const auto lb = lowerCaseOf(static_cast<unsigned char>(rhs[i]));

			if (la < lb)
			{
				return -1;
			}
			if (la > lb)
			{
				return 1;
			}
		}

		if (lhs.size() < rhs.size())
		{
			return -1;
		}
		if (lhs.size() > rhs.size())
		{
			return 1;
		}
		return 0;
	}

	template <typename StringLike, typename StringLikeToo>
	int CaseInsensitiveComparison(const StringLike& lhs, const StringLikeToo& rhs) noexcept
	{
		return CaseInsensitiveComparison(
		        static_cast<std::string_view>(lhs), static_cast<std::string_view>(rhs), ToLower);
	}

	template <typename StringLike, typename StringLikeToo>
	constexpr int CaseInsensitiveAsciiComparison(const StringLike& lhs, const StringLikeToo& rhs) noexcept
	{
		return CaseInsensitiveComparison(
		        static_cast<std::string_view>(lhs), static_cast<std::string_view>(rhs), AsciiToLower);
	}

	template <typename StringLike, typename StringLikeToo>
	bool IsCaseInsensitiveStringEqual(const StringLike& lhs, const StringLikeToo& rhs)
	{
		return CaseInsensitiveComparison(lhs, rhs) == 0;
	}

	template <typename StringLike, typename StringLikeToo>
	constexpr bool IsCaseInsensitiveAsciiStringEqual(const StringLike& lhs, const StringLikeToo& rhs)
	{
		return CaseInsensitiveAsciiComparison(lhs, rhs) == 0;
	}
}
