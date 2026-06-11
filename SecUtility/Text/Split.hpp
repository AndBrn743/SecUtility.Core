// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Diagnostic/TypeName.hpp>
#include <SecUtility/Misc/Bitflag.hpp>
#include <SecUtility/Raw/Int.hpp>
#include <charconv>
#include <string>
#include <string_view>
#include <vector>


namespace SecUtility
{
	template <typename T>
	struct ArithmeticParser
	{
		T operator()(const std::string_view value) const
		{
			T result{};

			const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), result);

			if (ec != std::errc{} || ptr != value.data() + value.size())
			{
				throw InvalidArgumentException("Failed to parse " + std::string{TypeName<T>});
			}

			return result;
		}
	};

	template <typename T>
	struct Parser;

	template <>
	struct Parser<std::string_view>
	{
		std::string_view operator()(const std::string_view value) const noexcept
		{
			// it's fine, caller explicitly asked for this
			// ReSharper disable once CppDFALocalValueEscapesFunction
			return value;
		}
	};

	template <>
	struct Parser<std::string>
	{
		std::string operator()(const std::string_view value) const
		{
			return std::string{value};
		}
	};

#define SEC_POPULATE_ARITHMETIC_PARSER(T)                                                                              \
	template <>                                                                                                        \
	struct Parser<T> : ArithmeticParser<T>                                                                             \
	{                                                                                                                  \
		/* NO CODE */                                                                                                  \
	};

	SEC_POPULATE_ARITHMETIC_PARSER(Int8)
	SEC_POPULATE_ARITHMETIC_PARSER(Int16)
	SEC_POPULATE_ARITHMETIC_PARSER(Int32)
	SEC_POPULATE_ARITHMETIC_PARSER(Int64)
	SEC_POPULATE_ARITHMETIC_PARSER(UInt8)
	SEC_POPULATE_ARITHMETIC_PARSER(UInt16)
	SEC_POPULATE_ARITHMETIC_PARSER(UInt32)
	SEC_POPULATE_ARITHMETIC_PARSER(UInt64)
	SEC_POPULATE_ARITHMETIC_PARSER(float)
	SEC_POPULATE_ARITHMETIC_PARSER(double)
	SEC_POPULATE_ARITHMETIC_PARSER(long double)

#undef SEC_POPULATE_ARITHMETIC_PARSER

	enum class SplitOptions
	{
		None = 0,
		SkipEmpty = 1 << 1,
		Trim = 1 << 2
	};

	template <>
	struct is_bitmask<SplitOptions> : std::true_type
	{
		/* NO CODE */
	};

	template <SplitOptions Options = {}, typename DelimiterPredicate, typename TParser = Parser<std::string>>
	auto Split(const std::string_view text, DelimiterPredicate isDelimiter, TParser parser = {})
	{
		std::vector<std::decay_t<decltype(parser(std::declval<std::string_view>()))>> result;

		const auto extractSubstr = [&text](std::size_t begin, std::size_t end)
		{
			if constexpr (static_cast<bool>(Options & SplitOptions::Trim))
			{
				while (begin < end && std::isspace(text[begin]))
				{
					++begin;
				}

				while (end > begin && std::isspace(text[end - 1]))
				{
					--end;
				}
			}

			return text.substr(begin, end - begin);
		};

		std::size_t tokenBegin = 0;

		for (std::size_t i = 0; i < text.size(); ++i)
		{
			if (!isDelimiter(text[i]))
			{
				continue;
			}

			if (const auto substr = extractSubstr(tokenBegin, i);
			    !(static_cast<bool>(Options & SplitOptions::SkipEmpty) && substr.empty()))
			{
				result.push_back(parser(substr));
			}

			tokenBegin = i + 1;
		}

		if (const auto substr = extractSubstr(tokenBegin, text.size());
		    !(static_cast<bool>(Options & SplitOptions::SkipEmpty) && substr.empty()))
		{
			result.push_back(parser(substr));
		}

		return result;
	}

	template <SplitOptions Options = {}, typename TParser = Parser<std::string>>
	auto Split(std::string_view text, char delimiter, TParser parser = {})
	{
		return Split<Options>(text, [delimiter](const char c) { return c == delimiter; }, parser);
	}
}
