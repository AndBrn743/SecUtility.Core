//
// Created by Andy on 2/19/2026.
//

#include <SecUtility/Diagnostic/Exception/Exception.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <string>
#include <string_view>


using namespace SecUtility;
using Catch::Matchers::Equals;


TEST_CASE("Exception - Default constructor")
{
	SECTION("Creates exception with generic message")
	{
		Exception ex;
		CHECK_THAT(ex.what(), Equals("Generic exception"));
	}
}


TEST_CASE("Exception - Single message constructor")
{
	SECTION("String literal message")
	{
		const Exception ex("Error occurred");
		CHECK_THAT(ex.what(), Equals("Error occurred"));
	}

	SECTION("String message")
	{
		const std::string msg = "Something went wrong";
		const Exception ex(msg);
		CHECK_THAT(ex.what(), Equals(msg));
	}

	SECTION("String_view message")
	{
		constexpr std::string_view msg = "View error";
		const Exception ex(msg);
		CHECK_THAT(ex.what(), Equals("View error"));
	}

	SECTION("Integer message")
	{
		const Exception ex(42);
		CHECK_THAT(ex.what(), Equals("42"));
	}

	SECTION("Double message")
	{
		const Exception ex(3.14);
		CHECK_THAT(ex.what(), Equals("3.140000")); // std::to_string format
	}

	SECTION("Mixed types - char*")
	{
		auto msg = "C-string error";
		const Exception ex(msg);
		CHECK_THAT(ex.what(), Equals("C-string error"));
	}
}


TEST_CASE("Exception - Multiple messages constructor")
{
	SECTION("Two string messages")
	{
		const Exception ex("First error", "Second error");
		CHECK_THAT(ex.what(), Equals("First error\n\t-> Second error"));
	}

	SECTION("Three string messages")
	{
		const Exception ex("Error 1", "Error 2", "Error 3");
		CHECK_THAT(ex.what(), Equals("Error 1\n\t-> Error 2\n\t-> Error 3"));
	}

	SECTION("String and integer")
	{
		const Exception ex("Error code:", 404);
		CHECK_THAT(ex.what(), Equals("Error code:\n\t-> 404"));
	}

	SECTION("Multiple types mixed")
	{
		const Exception ex("File:", "test.txt", "Line:", 42);
		CHECK_THAT(ex.what(), Equals("File:\n\t-> test.txt\n\t-> Line:\n\t-> 42"));
	}

	SECTION("Integer and string")
	{
		const Exception ex(500, "Internal server error");
		CHECK_THAT(ex.what(), Equals("500\n\t-> Internal server error"));
	}

	SECTION("Multiple numbers")
	{
		const Exception ex(1, 2, 3);
		CHECK_THAT(ex.what(), Equals("1\n\t-> 2\n\t-> 3"));
	}

	SECTION("Double and string")
	{
		const Exception ex("Value:", 3.14159, "is PI");
		CHECK_THAT(ex.what(), Equals("Value:\n\t-> 3.141590\n\t-> is PI"));
	}
}


TEST_CASE("Exception - Long messages")
{
	SECTION("Single long message")
	{
		const std::string longMsg(1000, 'A');
		const Exception ex(longMsg);
		CHECK_THAT(ex.what(), Equals(longMsg));
	}

	SECTION("Multiple long messages")
	{
		const std::string msg1(500, 'X');
		const std::string msg2(500, 'Y');
		const Exception ex(msg1, msg2);

		const std::string expected = msg1 + "\n\t-> " + msg2;
		CHECK_THAT(ex.what(), Equals(expected));
	}
}


TEST_CASE("Exception - Empty and special messages")
{
	SECTION("Empty string")
	{
		const Exception ex("");
		CHECK_THAT(ex.what(), Equals(""));
	}

	SECTION("Empty strings with other messages")
	{
		const Exception ex("", "Error", "");
		CHECK_THAT(ex.what(), Equals("\n\t-> Error\n\t-> "));
	}

	SECTION("Message with newlines")
	{
		const Exception ex("Line1\nLine2", "Line3");
		CHECK_THAT(ex.what(), Equals("Line1\nLine2\n\t-> Line3"));
	}

	SECTION("Message with tabs")
	{
		const Exception ex("Tab\there", "Tab\tthere");
		CHECK_THAT(ex.what(), Equals("Tab\there\n\t-> Tab\tthere"));
	}

	SECTION("Message with special characters")
	{
		const Exception ex("Special: !@#$%^&*()");
		CHECK_THAT(ex.what(), Equals("Special: !@#$%^&*()"));
	}
}


TEST_CASE("Exception - Move semantics")
{
	SECTION("Move constructor")
	{
		Exception ex1("Original message");
		const Exception ex2(std::move(ex1));
		CHECK_THAT(ex2.what(), Equals("Original message"));
	}

	SECTION("Move assignment")
	{
		Exception ex1("Message 1");
		Exception ex2("Message 2");
		ex2 = std::move(ex1);
		CHECK_THAT(ex2.what(), Equals("Message 1"));
	}
}


TEST_CASE("Exception - Throwing and catching")
{
	SECTION("Throw and catch by value")
	{
		const auto checkThrow = [] {
			throw Exception("Test exception");
		};

		CHECK_THROWS_WITH(checkThrow(), "Test exception");
	}

	SECTION("Throw and catch with multiple messages")
	{
		const auto checkThrow = [] {
			throw Exception("Error:", 404, "Not Found");
		};

		CHECK_THROWS_WITH(checkThrow(), "Error:\n\t-> 404\n\t-> Not Found");
	}

	SECTION("Catch as std::exception")
	{
		const auto checkThrow = [] {
			throw Exception("Standard exception");
		};

		CHECK_THROWS_WITH(checkThrow(), "Standard exception");
	}

	SECTION("Catch as const reference")
	{
		const auto checkThrow = [] {
			throw Exception("Const ref exception");
		};

		try
		{
			checkThrow();
		}
		catch (const Exception& ex)
		{
			CHECK_THAT(ex.what(), Equals("Const ref exception"));
		}
	}
}


TEST_CASE("Exception - Exception hierarchy")
{
	SECTION("Derived from std::exception")
	{
		Exception ex("Test");
		std::exception* base = &ex;
		REQUIRE(base != nullptr);
		CHECK_THAT(base->what(), Equals("Test"));
	}

	SECTION("Can catch as std::exception")
	{
		const auto checkThrow = [] {
			throw Exception("Derived exception");
		};

		try
		{
			checkThrow();
		}
		catch (const std::exception& ex)
		{
			CHECK_THAT(ex.what(), Equals("Derived exception"));
		}
	}
}


TEST_CASE("Exception - Numeric type conversions")
{
	SECTION("Integer types")
	{
		const Exception ex(123, 456L, static_cast<short>(78));
		CHECK_THAT(ex.what(), Equals("123\n\t-> 456\n\t-> 78"));
	}

	SECTION("Unsigned integer types")
	{
		const Exception ex(1U, 2UL);
		CHECK_THAT(ex.what(), Equals("1\n\t-> 2"));
	}

	SECTION("Floating point types")
	{
		const Exception ex(1.5F, 2.5);
		CHECK_THAT(ex.what(), Equals("1.500000\n\t-> 2.500000"));
	}

	SECTION("Boolean values")
	{
		const Exception ex(true, false);
		CHECK_THAT(ex.what(), Equals("1\n\t-> 0"));
	}
}


TEST_CASE("Exception - noexcept guarantees")
{
	SECTION("Constructor is noexcept")
	{
		const Exception ex("Test");
		// If compilation passes, noexcept is working
		// This is a compile-time check
		static_assert(noexcept(Exception("Test")), "Exception constructor should be noexcept");
		static_assert(noexcept(Exception(1, 2, 3)), "Exception constructor should be noexcept");
	}

	SECTION("what() is noexcept")
	{
		[[maybe_unused]] const Exception ex("Test");
		// This is a compile-time check
		static_assert(noexcept(ex.what()), "what() should be noexcept");
	}
}


TEST_CASE("Exception - Message joiner format")
{
	SECTION("Verifies joiner format")
	{
		const Exception ex("A", "B", "C");
		const std::string msg = ex.what();

		// Check for newline and tab-> pattern
		CHECK(msg.find("\n\t-> ") != std::string::npos);
	}

	SECTION("Single message has no joiner")
	{
		const Exception ex("Single");
		const std::string msg = ex.what();

		CHECK(msg.find("\n\t-> ") == std::string::npos);
		CHECK_THAT(msg, Equals("Single"));
	}
}


TEST_CASE("Exception - what() returns const char*")
{
	SECTION("Pointer remains valid")
	{
		Exception ex("Persistent message");
		const char* what1 = ex.what();
		const char* what2 = ex.what();

		// Pointers should be the same (internal c_str() pointer)
		CHECK(what1 == what2);
	}

	SECTION("Content is null-terminated")
	{
		const Exception ex("Test\0Hidden", 42);
		const std::string msg = ex.what();

		CHECK(msg.length() > 0);
	}
}
