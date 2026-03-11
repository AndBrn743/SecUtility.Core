// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <string>
#include <string_view>

#if !defined(_WIN32) && __has_include(<cxxabi.h>)
#include <cxxabi.h>
#endif


namespace SecUtility
{
	namespace Detail
	{
		template <typename T /* do not remove or rename `T`*/>
		constexpr std::string_view TypeName()
		{
			// clang-format off
#if defined(__clang__)
			std::string_view name = __PRETTY_FUNCTION__;
			constexpr auto prefixSize = std::string_view{"std::string_view SecUtility::Detail::TypeName() [T = "}.size();
			constexpr auto suffixSize = std::string_view{"]"}.size();
#elif defined(__GNUC__)
			std::string_view name = __PRETTY_FUNCTION__;
			constexpr auto prefixSize = std::string_view{"constexpr std::string_view SecUtility::Detail::TypeName() [with T = "}.size();
			constexpr auto suffixSize = std::string_view{"; std::string_view = std::basic_string_view<char>]"}.size();
#elif defined(_MSC_VER)
			std::string_view name = __FUNCSIG__;
			constexpr auto prefixSize = std::string_view{"class std::basic_string_view<char,struct std::char_traits<char> > __cdecl SecUtility::Detail::TypeName<"}.size();
			constexpr auto suffixSize = std::string_view{">(void)"}.size();
#endif
			// clang-format on

			name.remove_prefix(prefixSize);
			name.remove_suffix(suffixSize);
			// ReSharper disable once CppDFALocalValueEscapesFunction
			return name;  // it's fine, since `name` is referencing immortal
		}
	}


	template <typename T>
	inline constexpr std::string_view TypeName = Detail::TypeName<T>();


	template <class T>
	constexpr std::string_view TypeNameOf(T&&)
	{
		return Detail::TypeName<T>();
	}


	inline std::string Demangle(const std::string_view& mangledName)
	{
#if !defined(_WIN32) && __has_include(<cxxabi.h>)
		int status;
		char* name = abi::__cxa_demangle(mangledName.begin(), nullptr, nullptr, &status);
		if (status != 0)
		{
			return std::string{mangledName};
		}

		std::string demangledName = name;
		std::free(name);
		return demangledName;
#else
		return std::string{mangledName};
#endif
	}
}
