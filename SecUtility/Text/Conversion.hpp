// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Diagnostic/TypeName.hpp>
#include <SecUtility/Raw/Int.hpp>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

#if __has_include(<nlohmann/json.hpp>)
#include <nlohmann/json.hpp>
#endif

namespace SecUtility
{
	namespace Detail::Conversion
	{
		template <typename T>
		T ParseArithmetic(const std::string_view value)
		{
			// strto* requires a NUL-terminated input, while string_view does not provide one.
			const std::string text{value};
			char* end{};
			errno = 0;

			// Match from_chars' stricter token grammar: no leading whitespace or plus sign.
			if (text.empty() || text.front() == '+' || text.front() == ' ' || text.front() == '\t'
			    || text.front() == '\n' || text.front() == '\r' || text.front() == '\f' || text.front() == '\v')
			{
				throw InvalidArgumentException("Failed to parse " + std::string{::SecUtility::TypeName<T>});
			}

			if constexpr (std::is_integral_v<T> && std::is_signed_v<T>)
			{
				const auto result = std::strtoll(text.c_str(), &end, 10);
				if (errno == ERANGE || end != text.c_str() + text.size() || result < std::numeric_limits<T>::min()
				    || result > std::numeric_limits<T>::max())
				{
					throw InvalidArgumentException("Failed to parse " + std::string{::SecUtility::TypeName<T>});
				}
				return static_cast<T>(result);
			}
			else if constexpr (std::is_integral_v<T>)
			{
				const auto result = std::strtoull(text.c_str(), &end, 10);
				if (text.front() == '-' || errno == ERANGE || end != text.c_str() + text.size()
				    || result > std::numeric_limits<T>::max())
				{
					throw InvalidArgumentException("Failed to parse " + std::string{::SecUtility::TypeName<T>});
				}
				return static_cast<T>(result);
			}
			else if constexpr (std::is_same_v<T, float>)
			{
				const auto result = std::strtof(text.c_str(), &end);
				if (errno == ERANGE || end != text.c_str() + text.size())
				{
					throw InvalidArgumentException("Failed to parse " + std::string{::SecUtility::TypeName<T>});
				}
				return result;
			}
			else if constexpr (std::is_same_v<T, double>)
			{
				const auto result = std::strtod(text.c_str(), &end);
				if (errno == ERANGE || end != text.c_str() + text.size())
				{
					throw InvalidArgumentException("Failed to parse " + std::string{::SecUtility::TypeName<T>});
				}
				return result;
			}
			else
			{
				static_assert(std::is_same_v<T, long double>);
				const auto result = std::strtold(text.c_str(), &end);
				if (errno == ERANGE || end != text.c_str() + text.size())
				{
					throw InvalidArgumentException("Failed to parse " + std::string{::SecUtility::TypeName<T>});
				}
				return result;
			}
		}
	}

	template <typename T>
	struct ArithmeticParser
	{
		T operator()(const std::string_view value) const
		{
			return Detail::Conversion::ParseArithmetic<T>(value);
		}
	};

	template <typename T>
	struct Parser : ArithmeticParser<T>
	{
		/* NO CODE */
	};

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

	template <>
	struct Parser<bool>
	{
		bool operator()(const std::string_view value) const
		{
			if (value == "0" || value == "false")
			{
				return false;
			}

			if (value == "1" || value == "true")
			{
				return true;
			}

			throw InvalidOperationException("Can't parse \"" + std::string{value} + "\" to bool");
		}
	};

	template <typename T>
	T Parse(const std::string_view string)
	{
		return Parser<T>{}(string);
	}

	template <typename T>
	T Parse(const std::string& string)
	{
		return Parse<T>(std::string_view{string});
	}

#if defined(NLOHMANN_JSON_VERSION_MAJOR)
	template <typename T>
	T Parse(const nlohmann::json& json)
	{
		return json.get<T>();
	}

	template <>
	inline double Parse(const nlohmann::json& json)
	{
		return Parse<double>(json.get<std::string>());
	}

	template <>
	inline float Parse(const nlohmann::json& json)
	{
		return Parse<float>(json.get<std::string>());
	}

	template <>
	inline long double Parse(const nlohmann::json& json)
	{
		return Parse<long double>(json.get<std::string>());
	}
#endif

	namespace Detail::ParseTo
	{
		template <typename Result>
		struct ParseToFunctor
		{
			template <typename Input>
			friend Result operator|(const Input& input, ParseToFunctor) noexcept
			{
				return Parse<Result>(input);
			}
		};
	}

	template <typename Result>
	inline constexpr Detail::ParseTo::ParseToFunctor<Result> ParseTo;
}
