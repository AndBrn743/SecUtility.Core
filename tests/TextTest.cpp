// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#include <SecUtility/Text/CaseConversion.hpp>
#include <SecUtility/Text/Split.hpp>
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

TEST_CASE("String split")
{
	SECTION("Split on char delimiter")
	{
		SECTION("Basic split")
		{
			const auto parts = Split("a,b,c", ',');
			REQUIRE(parts.size() == 3);
			CHECK_THAT(parts[0], Equals("a"));
			CHECK_THAT(parts[1], Equals("b"));
			CHECK_THAT(parts[2], Equals("c"));
		}

		SECTION("Single element (no delimiter)")
		{
			const auto parts = Split("hello", ',');
			REQUIRE(parts.size() == 1);
			CHECK_THAT(parts[0], Equals("hello"));
		}

		SECTION("Empty string")
		{
			const auto parts = Split("", ',');
			REQUIRE(parts.size() == 1);
			CHECK_THAT(parts[0], Equals(""));
		}

		SECTION("Only delimiter")
		{
			const auto parts = Split(",", ',');
			REQUIRE(parts.size() == 2);
			CHECK_THAT(parts[0], Equals(""));
			CHECK_THAT(parts[1], Equals(""));
		}

		SECTION("Leading delimiter")
		{
			const auto parts = Split(",a,b", ',');
			REQUIRE(parts.size() == 3);
			CHECK_THAT(parts[0], Equals(""));
			CHECK_THAT(parts[1], Equals("a"));
			CHECK_THAT(parts[2], Equals("b"));
		}

		SECTION("Trailing delimiter")
		{
			const auto parts = Split("a,b,", ',');
			REQUIRE(parts.size() == 3);
			CHECK_THAT(parts[0], Equals("a"));
			CHECK_THAT(parts[1], Equals("b"));
			CHECK_THAT(parts[2], Equals(""));
		}

		SECTION("Consecutive delimiters")
		{
			const auto parts = Split("a,,b", ',');
			REQUIRE(parts.size() == 3);
			CHECK_THAT(parts[0], Equals("a"));
			CHECK_THAT(parts[1], Equals(""));
			CHECK_THAT(parts[2], Equals("b"));
		}

		SECTION("Empties can be removed with SplitOptions::SkipEmpty")
		{
			const auto parts = Split<SplitOptions::SkipEmpty>("a,,b", ',');
			REQUIRE(parts.size() == 2);
			CHECK_THAT(parts[0], Equals("a"));
			CHECK_THAT(parts[1], Equals("b"));
		}

		SECTION("Space delimiter")
		{
			const auto parts = Split("hello world", ' ');
			REQUIRE(parts.size() == 2);
			CHECK_THAT(parts[0], Equals("hello"));
			CHECK_THAT(parts[1], Equals("world"));
		}
	}

	SECTION("Split with predicate")
	{
		SECTION("Whitespace predicate")
		{
			const auto parts = Split("a b\tc\nd", [](const char c) { return c == ' ' || c == '\t' || c == '\n'; });
			REQUIRE(parts.size() == 4);
			CHECK_THAT(parts[0], Equals("a"));
			CHECK_THAT(parts[1], Equals("b"));
			CHECK_THAT(parts[2], Equals("c"));
			CHECK_THAT(parts[3], Equals("d"));
		}

		SECTION("Predicate that matches nothing")
		{
			const auto parts = Split("abc", [](const char) { return false; });
			REQUIRE(parts.size() == 1);
			CHECK_THAT(parts[0], Equals("abc"));
		}

		SECTION("Predicate that matches everything")
		{
			const auto parts = Split("abc", [](const char) { return true; });
			REQUIRE(parts.size() == 4);
			CHECK_THAT(parts[0], Equals(""));
			CHECK_THAT(parts[1], Equals(""));
			CHECK_THAT(parts[2], Equals(""));
			CHECK_THAT(parts[3], Equals(""));
		}
	}

	SECTION("Split with string_view parser")
	{
		const auto parts = Split("a,b,c", ',', Parser<std::string_view>{});
		REQUIRE(parts.size() == 3);
		CHECK(parts[0] == "a");
		CHECK(parts[1] == "b");
		CHECK(parts[2] == "c");
	}

	SECTION("Split with integer parser")
	{
		SECTION("Parse integers")
		{
			const auto parts = Split("1,2,3", ',', Parser<Int32>{});
			REQUIRE(parts.size() == 3);
			CHECK(parts[0] == 1);
			CHECK(parts[1] == 2);
			CHECK(parts[2] == 3);
		}

		SECTION("Parse negative integers")
		{
			const auto parts = Split("-10,0,10", ',', Parser<Int32>{});
			REQUIRE(parts.size() == 3);
			CHECK(parts[0] == -10);
			CHECK(parts[1] == 0);
			CHECK(parts[2] == 10);
		}

		SECTION("Invalid integer throws InvalidArgumentException")
		{
			CHECK_THROWS_AS(Split("1,abc,3", ',', Parser<Int32>{}), InvalidArgumentException);
		}

		SECTION("Partial parse failure throws")
		{
			CHECK_THROWS_AS(Split("123abc", ',', Parser<Int32>{}), InvalidArgumentException);
		}

		SECTION("Empty token throws for integer parse")
		{
			CHECK_THROWS_AS(Split("1,,3", ',', Parser<Int32>{}), InvalidArgumentException);
		}
	}

	SECTION("Split with floating-point parser")
	{
		SECTION("Parse doubles")
		{
			const auto parts = Split("1.5,2.5,3.5", ',', Parser<double>{});
			REQUIRE(parts.size() == 3);
			CHECK(parts[0] == 1.5);
			CHECK(parts[1] == 2.5);
			CHECK(parts[2] == 3.5);
		}

		SECTION("Invalid double throws InvalidArgumentException")
		{
			CHECK_THROWS_AS(Split("1.0,not_a_number", ',', Parser<double>{}), InvalidArgumentException);
		}
	}

	SECTION("Split preserves original string_view lifetime")
	{
		const std::string original = "x,y,z";
		const auto parts = Split(original, ',', Parser<std::string_view>{});
		REQUIRE(parts.size() == 3);
		CHECK(parts[0] == "x");
		CHECK(parts[1] == "y");
		CHECK(parts[2] == "z");
	}

	SECTION("SplitOptions::Trim works")
	{
		const std::string original = " x, y ,    ,z ";
		{
			const auto parts = Split(original, ',', Parser<std::string_view>{});
			REQUIRE(parts.size() == 4);
			CHECK(parts[0] == " x");
			CHECK(parts[1] == " y ");
			CHECK(parts[2] == "    ");
			CHECK(parts[3] == "z ");
		}
		{
			const auto parts = Split<SplitOptions::Trim>(original, ',', Parser<std::string_view>{});
			REQUIRE(parts.size() == 4);
			CHECK(parts[0] == "x");
			CHECK(parts[1] == "y");
			CHECK(parts[2] == "");
			CHECK(parts[3] == "z");
		}
		{
			const auto parts =
			        Split<SplitOptions::Trim | SplitOptions::SkipEmpty>(original, ',', Parser<std::string_view>{});
			REQUIRE(parts.size() == 3);
			CHECK(parts[0] == "x");
			CHECK(parts[1] == "y");
			CHECK(parts[2] == "z");
		}
	}

	SECTION("Split respects quoted delimiters")
	{
		auto result = Split<SplitOptions::EnableQuotes>(R"(a,"b,c",d)", ',');

		REQUIRE(result.size() == 3);
		REQUIRE(result[0] == "a");
		REQUIRE(result[1] == "\"b,c\"");
		REQUIRE(result[2] == "d");
	}

	// -------------------------
	// empty quoted token
	// -------------------------
	SECTION("Split preserves empty quoted token")
	{
		auto result = Split<SplitOptions::EnableQuotes | SplitOptions::SkipEmpty>(R"(a,"",b)", ',');

		REQUIRE(result.size() == 3);
		REQUIRE(result[0] == "a");
		REQUIRE(result[1] == R"("")");
		REQUIRE(result[2] == "b");
	}

	// -------------------------
	// escape sequences inside quotes
	// -------------------------
	SECTION("Split handles escaped quotes inside quoted strings")
	{
		auto result = Split<SplitOptions::EnableQuotes>(R"(a,"b\"c",d)", ',');

		REQUIRE(result.size() == 3);
		REQUIRE(result[1] == R"("b\"c")");
	}

	// -------------------------
	// backslash literal inside quotes
	// -------------------------
	SECTION("Split handles escaped backslash inside quotes")
	{
		auto result = Split<SplitOptions::EnableQuotes>(R"(a,"b,\\,c",d)", ',');

		REQUIRE(result.size() == 3);
		REQUIRE(result[1] == R"("b,\\,c")");
	}

	// -------------------------
	// unclosed quote throws
	// -------------------------
	SECTION("Split throws on unterminated quote")
	{
		{
			REQUIRE_THROWS_AS(Split<SplitOptions::EnableQuotes>(R"(a,"b,c)", ','), std::runtime_error);
		}
		{
			REQUIRE_NOTHROW(Split(R"(a,"b,c)", ','));
		}
	}

	// -------------------------
	// whitespace outside quotes is trimmed only if enabled
	// -------------------------
	SECTION("Trim does not affect quoted content")
	{
		auto result = Split<SplitOptions::EnableQuotes | SplitOptions::Trim>(R"(  a ,  "  b, c  " , d  )", ',');

		REQUIRE(result.size() == 3);
		REQUIRE(result[0] == "a");
		REQUIRE(result[1] == R"("  b, c  ")");
		REQUIRE(result[2] == "d");
	}
}
