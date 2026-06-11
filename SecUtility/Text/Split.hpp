// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Diagnostic/TypeName.hpp>
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

	template <typename DelimiterPredicate, typename TParser = Parser<std::string>>
	auto Split(const std::string_view text, DelimiterPredicate isDelimiter, TParser parser = {})
	{
		std::vector<std::decay_t<decltype(parser(std::declval<std::string_view>()))>> result;

		std::size_t tokenBegin = 0;

		for (std::size_t i = 0; i < text.size(); ++i)
		{
			if (!isDelimiter(text[i]))
			{
				continue;
			}

			result.push_back(parser(text.substr(tokenBegin, i - tokenBegin)));

			tokenBegin = i + 1;
		}

		result.push_back(parser(text.substr(tokenBegin)));

		return result;
	}

	template <typename TParser = Parser<std::string>>
	auto Split(std::string_view text, char delimiter, TParser parser = {})
	{
		return Split(text, [delimiter](const char c) { return c == delimiter; }, parser);
	}
}
