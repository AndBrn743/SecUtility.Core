//
// Created by Andy on 3/11/2026.
//

#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <regex>

#include <SecUtility/Diagnostic/TypeName.hpp>


static bool Matched(const std::string_view sv, const std::regex& regex)
{
	std::cmatch m;
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

	REQUIRE((SecUtility::TypeName<long const volatile long> == "volatile const long long"
	         || SecUtility::TypeName<long const volatile long> == "const volatile long long"
	         || SecUtility::TypeName<long const volatile long> == "volatile const long long int"
	         || SecUtility::TypeName<long const volatile long> == "const volatile long long int"));
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
	STATIC_REQUIRE(SecUtility::TypeName<Example::Internal::Array<int, 3>>
	               == std::string_view{"Example::Internal::Array<int, 3>"});

	STATIC_REQUIRE(SecUtility::TypeName<const Example::Internal::Array<int, 4>>
	               == std::string_view{"const Example::Internal::Array<int, 4>"});

	REQUIRE(Matched(SecUtility::TypeName<Example::Internal::Array<int, 5>&>,
	                std::regex{R"(^Example::Internal::Array<int, 5>\s*&$)"}));

	REQUIRE(Matched(SecUtility::TypeName<const Example::Internal::Array<int, 6>&>,
	                std::regex{R"(^const Example::Internal::Array<int, 6>\s*&$)"}));

	REQUIRE(Matched(SecUtility::TypeName<Example::Internal::Array<int, 7>&&>,
	                std::regex{R"(^Example::Internal::Array<int, 7>\s*&&$)"}));

	REQUIRE(Matched(SecUtility::TypeName<const Example::Internal::Array<int, 42>&&>,
	                std::regex{R"(^const Example::Internal::Array<int, 42>\s*&&$)"}));
}

#include <typeinfo>

TEST_CASE("Demangle should work")
{
	REQUIRE(SecUtility::Demangle(typeid(Example::Internal::Array<int, 3>).name())
	        == std::string_view{"Example::Internal::Array<int, 3ul>"});
}
