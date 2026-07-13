// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Misc/Bitflag.hpp>
#include <SecUtility/Text/Conversion.hpp>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>


namespace SecUtility
{
	enum class SplitOptions
	{
		None = 0,
		SkipEmpty = 1 << 1,
		Trim = 1 << 2,
		EnableQuotes = 1 << 3
	};

	template <>
	struct is_bitmask<SplitOptions> : std::true_type
	{
		/* NO CODE */
	};

	template <SplitOptions Options = SplitOptions{},
	          typename DelimiterPredicate,
	          typename TParser = Parser<std::string>>
	auto Split(const std::string_view text, DelimiterPredicate isDelimiter, TParser parser = {})
	{
		std::vector<std::decay_t<decltype(parser(std::declval<std::string_view>()))>> result;

		bool isInQuotes = false;
		bool isEscapeNext = false;

		const auto extractSubstr = [&text](std::size_t begin, std::size_t end)
		{
			if constexpr (static_cast<bool>(Options & SplitOptions::Trim))
			{
				while (begin < end && std::isspace(static_cast<unsigned char>(text[begin])))
				{
					++begin;
				}

				while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])))
				{
					--end;
				}
			}

			return text.substr(begin, end - begin);
		};

		std::size_t tokenBegin = 0;

		for (std::size_t i = 0; i < text.size(); ++i)
		{
			const char c = text[i];

			// --- quote / escape handling ---
			if constexpr (static_cast<bool>(Options & SplitOptions::EnableQuotes))
			{
				if (isInQuotes)
				{
					if (isEscapeNext)
					{
						isEscapeNext = false;
					}
					else if (c == '\\')
					{
						isEscapeNext = true;
						continue;
					}
					else if (c == '"')
					{
						isInQuotes = false;
						continue;
					}
				}
				else
				{
					if (c == '"')
					{
						isInQuotes = true;
						continue;
					}
				}
			}

			// --- delimiter handling (only outside quotes) ---
			if (!isInQuotes && isDelimiter(c))
			{

				if (const auto substr = extractSubstr(tokenBegin, i);
				    !(static_cast<bool>(Options & SplitOptions::SkipEmpty) && substr.empty()))
				{
					result.push_back(parser(substr));
				}

				tokenBegin = i + 1;
			}
		}

		// --- final token ---
		if constexpr (static_cast<bool>(Options & SplitOptions::EnableQuotes))
		{
			if (isInQuotes)
			{
				throw std::runtime_error("Split: unterminated quote in input");
			}
		}

		if (const auto substr = extractSubstr(tokenBegin, text.size());
		    !(static_cast<bool>(Options & SplitOptions::SkipEmpty) && substr.empty()))
		{
			result.push_back(parser(substr));
		}

		return result;
	}

	template <SplitOptions Options = SplitOptions{}, typename TParser = Parser<std::string>>
	auto Split(std::string_view text, char delimiter, TParser parser = {})
	{
		return Split<Options>(text, [delimiter](const char c) { return c == delimiter; }, parser);
	}
}
