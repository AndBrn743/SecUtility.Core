//
// Created by Andy on 3/11/2026.
//

#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <regex>

#include <SecUtility/Diagnostic/TypeName.hpp>


static bool Matched(const std::string_view sv, const std::regex& regex)
{
	std::match_results<std::string_view::const_iterator> m;
	return std::regex_match(sv.cbegin(), sv.cend(), m, regex);
}

TEST_CASE("Constexpr TypeName should work with builtins")
{
	STATIC_REQUIRE(SecUtility::TypeName<int> == std::string_view{"int"});

	STATIC_REQUIRE(SecUtility::TypeName<const int> == std::string_view{"const int"});

	REQUIRE(Matched(SecUtility::TypeName<int&>, std::regex{R"(^int\s*&$)"}));

	REQUIRE(Matched(SecUtility::TypeName<const int&>, std::regex{R"(^const int\s*&$)"}));

	REQUIRE(Matched(SecUtility::TypeName<int&&>, std::regex{R"(^int\s*&&$)"}));

	REQUIRE(Matched(SecUtility::TypeName<const int&&>, std::regex{R"(^const int\s*&&$)"}));

	// what is MSVC doing?
	REQUIRE((SecUtility::TypeName<long const volatile long> == "volatile const long long"
	         || SecUtility::TypeName<long const volatile long> == "const volatile long long"
	         || SecUtility::TypeName<long const volatile long> == "volatile const long long int"
	         || SecUtility::TypeName<long const volatile long> == "const volatile long long int"
	         || SecUtility::TypeName<long const volatile long> == "volatile const __int64"));
}

namespace Example::Internal
{
	template <typename T, std::size_t S>
	class Array
	{
		/* NO CODE */
	};
}

TEST_CASE("Constexpr TypeName should work with user defined")
{
#if defined(_MSC_VER)
	STATIC_REQUIRE(SecUtility::TypeName<Example::Internal::Array<int, 3>>
	               == std::string_view{"class Example::Internal::Array<int,3>"});

	STATIC_REQUIRE(SecUtility::TypeName<const Example::Internal::Array<int, 4>>
	               == std::string_view{"const class Example::Internal::Array<int,4>"});
#else
	STATIC_REQUIRE(SecUtility::TypeName<Example::Internal::Array<int, 3>>
	               == std::string_view{"Example::Internal::Array<int, 3>"});

	STATIC_REQUIRE(SecUtility::TypeName<const Example::Internal::Array<int, 4>>
	               == std::string_view{"const Example::Internal::Array<int, 4>"});
#endif

	REQUIRE(Matched(SecUtility::TypeName<Example::Internal::Array<int, 5>&>,
	                std::regex{R"(^(class )?Example::Internal::Array<int,\s?5>\s*&$)"}));

	REQUIRE(Matched(SecUtility::TypeName<const Example::Internal::Array<int, 6>&>,
	                std::regex{R"(^const (class )?Example::Internal::Array<int,\s?6>\s*&$)"}));

	REQUIRE(Matched(SecUtility::TypeName<Example::Internal::Array<int, 7>&&>,
	                std::regex{R"(^(class )?Example::Internal::Array<int,\s?7>\s*&&$)"}));

	REQUIRE(Matched(SecUtility::TypeName<const Example::Internal::Array<int, 42>&&>,
	                std::regex{R"(^const (class )?Example::Internal::Array<int,\s?42>\s*&&$)"}));
}

#include <typeinfo>

TEST_CASE("Demangle should work")
{
#if defined(_MSC_VER)
	REQUIRE(SecUtility::Demangle(typeid(Example::Internal::Array<int, 3>).name())
	        == std::string_view{"class Example::Internal::Array<int,3>"});
#else
	REQUIRE(SecUtility::Demangle(typeid(Example::Internal::Array<int, 3>).name())
	        == std::string_view{"Example::Internal::Array<int, 3ul>"});
#endif
}
