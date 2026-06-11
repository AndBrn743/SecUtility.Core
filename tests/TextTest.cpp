// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#include <SecUtility/Text/CaseConversion.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <string>
#include <utility>


using namespace SecUtility;
using Catch::Matchers::Equals;


TEST_CASE("Case conversion tests")
{
	SECTION("ToLower converts single characters")
	{
		SECTION("Uppercase to lowercase")
		{
			CHECK(ToLower('A') == 'a');
			CHECK(ToLower('Z') == 'z');
			CHECK(ToLower('M') == 'm');
		}

		SECTION("Already lowercase is unchanged")
		{
			CHECK(ToLower('a') == 'a');
			CHECK(ToLower('z') == 'z');
		}

		SECTION("Non-alpha is unchanged")
		{
			CHECK(ToLower('0') == '0');
			CHECK(ToLower('9') == '9');
			CHECK(ToLower(' ') == ' ');
			CHECK(ToLower('!') == '!');
		}
	}

	SECTION("ToLower converts strings by value")
	{
		CHECK_THAT(ToLower(std::string{"Hello World 123"}), Equals("hello world 123"));
		CHECK_THAT(ToLower(std::string{"ALL CAPS"}), Equals("all caps"));
		CHECK_THAT(ToLower(std::string{"already lower"}), Equals("already lower"));
		CHECK_THAT(ToLower(std::string{""}), Equals(""));
		CHECK_THAT(ToLower(std::string{"123!@#"}), Equals("123!@#"));
	}

	SECTION("ToLower converts strings in place")
	{
		SECTION("Mixed case")
		{
			std::string s = "Hello World";
			ToLower(s, std::in_place);
			CHECK_THAT(s, Equals("hello world"));
		}

		SECTION("Empty string")
		{
			std::string s;
			ToLower(s, std::in_place);
			CHECK_THAT(s, Equals(""));
		}

		SECTION("No alpha characters")
		{
			std::string s = "12345";
			ToLower(s, std::in_place);
			CHECK_THAT(s, Equals("12345"));
		}
	}

	SECTION("ToUpper converts single characters")
	{
		SECTION("Lowercase to uppercase")
		{
			CHECK(ToUpper('a') == 'A');
			CHECK(ToUpper('z') == 'Z');
			CHECK(ToUpper('m') == 'M');
		}

		SECTION("Already uppercase is unchanged")
		{
			CHECK(ToUpper('A') == 'A');
			CHECK(ToUpper('Z') == 'Z');
		}

		SECTION("Non-alpha is unchanged")
		{
			CHECK(ToUpper('0') == '0');
			CHECK(ToUpper(' ') == ' ');
		}
	}

	SECTION("ToUpper converts strings by value")
	{
		CHECK_THAT(ToUpper(std::string{"Hello World 123"}), Equals("HELLO WORLD 123"));
		CHECK_THAT(ToUpper(std::string{"all lower"}), Equals("ALL LOWER"));
		CHECK_THAT(ToUpper(std::string{""}), Equals(""));
	}

	SECTION("ToUpper converts strings in place")
	{
		std::string s = "Hello World";
		ToUpper(s, std::in_place);
		CHECK_THAT(s, Equals("HELLO WORLD"));
	}

	SECTION("AsciiToLower converts single characters")
	{
		CHECK(AsciiToLower('A') == 'a');
		CHECK(AsciiToLower('Z') == 'z');
		CHECK(AsciiToLower('a') == 'a');
		CHECK(AsciiToLower('0') == '0');
		CHECK(AsciiToLower(' ') == ' ');
	}

	SECTION("AsciiToLower converts strings by value")
	{
		CHECK_THAT(AsciiToLower(std::string{"Hello World"}), Equals("hello world"));
		CHECK_THAT(AsciiToLower(std::string{""}), Equals(""));
		CHECK_THAT(AsciiToLower(std::string{"NOCHANGE123"}), Equals("nochange123"));
	}

	SECTION("AsciiToLower converts strings in place")
	{
		std::string s = "Hello World";
		AsciiToLower(s, std::in_place);
		CHECK_THAT(s, Equals("hello world"));
	}

	SECTION("AsciiToUpper converts single characters")
	{
		CHECK(AsciiToUpper('a') == 'A');
		CHECK(AsciiToUpper('z') == 'Z');
		CHECK(AsciiToUpper('A') == 'A');
		CHECK(AsciiToUpper('0') == '0');
		CHECK(AsciiToUpper(' ') == ' ');
	}

	SECTION("AsciiToUpper converts strings by value")
	{
		CHECK_THAT(AsciiToUpper(std::string{"Hello World"}), Equals("HELLO WORLD"));
		CHECK_THAT(AsciiToUpper(std::string{""}), Equals(""));
		CHECK_THAT(AsciiToUpper(std::string{"nochange123"}), Equals("NOCHANGE123"));
	}

	SECTION("AsciiToUpper converts strings in place")
	{
		std::string s = "Hello World";
		AsciiToUpper(s, std::in_place);
		CHECK_THAT(s, Equals("HELLO WORLD"));
	}

	SECTION("By-value string conversion does not modify the original")
	{
		std::string original = "Hello";
		const std::string copy = original;

		SECTION("ToLower")
		{
			auto result = ToLower(original);
			CHECK_THAT(original, Equals(copy));
			CHECK_THAT(result, Equals("hello"));
		}

		SECTION("ToUpper")
		{
			auto result = ToUpper(original);
			CHECK_THAT(original, Equals(copy));
			CHECK_THAT(result, Equals("HELLO"));
		}

		SECTION("AsciiToLower")
		{
			auto result = AsciiToLower(original);
			CHECK_THAT(original, Equals(copy));
			CHECK_THAT(result, Equals("hello"));
		}

		SECTION("AsciiToUpper")
		{
			auto result = AsciiToUpper(original);
			CHECK_THAT(original, Equals(copy));
			CHECK_THAT(result, Equals("HELLO"));
		}
	}
}
