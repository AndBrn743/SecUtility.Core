// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <string>

#if __has_include(<nlohmann/json.hpp>)
#include <nlohmann/json.hpp>
#endif

namespace SecUtility
{
	template <typename T>
	T Parse(const std::string& string);

	template <>
	inline int Parse(const std::string& string)
	{
		return std::stoi(string);
	}

	template <>
	inline long Parse(const std::string& string)
	{
		return std::stol(string);
	}

	template <>
	inline unsigned long Parse(const std::string& string)
	{
		return std::stoul(string);
	}

	template <>
	inline long long Parse(const std::string& string)
	{
		return std::stoll(string);
	}

	template <>
	inline unsigned long long Parse(const std::string& string)
	{
		return std::stoull(string);
	}

	template <>
	inline double Parse(const std::string& string)
	{
		return std::stod(string);
	}

	template <>
	inline float Parse(const std::string& string)
	{
		return std::stof(string);
	}

	template <>
	inline long double Parse(const std::string& string)
	{
		return std::stold(string);
	}

	template <>
	inline bool Parse(const std::string& string)
	{
		if (string == "0" || string == "false")
		{
			return false;
		}

		if (string == "1" || string == "true")
		{
			return true;
		}

		throw InvalidOperationException("Can't parse \"" + string + "\" to bool");
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
		return std::stod(json.get<std::string>());
	}

	template <>
	inline float Parse(const nlohmann::json& json)
	{
		return std::stof(json.get<std::string>());
	}

	template <>
	inline long double Parse(const nlohmann::json& json)
	{
		return std::stold(json.get<std::string>());
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
