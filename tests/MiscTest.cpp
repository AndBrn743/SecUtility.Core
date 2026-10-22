//
// Created by Andy on 7/11/2026.
//

#include <SecUtility/Misc/RefCountedString.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

using namespace SecUtility;
using Catch::Matchers::Equals;


TEST_CASE("RefCountedString - Construction")
{
	SECTION("Default constructed string is empty")
	{
		RefCountedString s;
		CHECK(s.Size() == 0);
		CHECK(s.CString() != nullptr);
		CHECK_THAT(s.CString(), Equals(""));
	}

	SECTION("Construct from const char*")
	{
		const RefCountedString s = "hello";
		CHECK(s.Size() == 5);
		CHECK_THAT(s.CString(), Equals("hello"));
	}

	SECTION("Construct from std::string")
	{
		const RefCountedString s = std::string("world");
		CHECK(s.Size() == 5);
		CHECK_THAT(s.CString(), Equals("world"));
	}

	SECTION("Construct from std::string_view")
	{
		const RefCountedString s = std::string_view{"view!"};
		CHECK(s.Size() == 5);
		CHECK_THAT(s.CString(), Equals("view!"));
	}

	SECTION("Construct from empty string literal")
	{
		const RefCountedString s = "";
		CHECK(s.Size() == 0);
		CHECK_THAT(s.CString(), Equals(""));
	}

	SECTION("Construct from long string (exceeds SSO)")
	{
		const std::string long_str(1000, 'x');
		const RefCountedString s = long_str;
		CHECK(s.Size() == 1000);
		CHECK_THAT(s.CString(), Equals(long_str));
	}
}


TEST_CASE("RefCountedString - Element access")
{
	const RefCountedString s = "abcde";

	SECTION("operator[] returns each character")
	{
		CHECK(s[0] == 'a');
		CHECK(s[1] == 'b');
		CHECK(s[2] == 'c');
		CHECK(s[3] == 'd');
		CHECK(s[4] == 'e');
	}

	SECTION("CString is null-terminated")
	{
		CHECK(s.CString()[s.Size()] == '\0');
	}

	SECTION("Length matches Size")
	{
		CHECK(s.Length() == s.Size());
	}
}


TEST_CASE("RefCountedString - Copy semantics")
{
	SECTION("Copy constructed string equals original")
	{
		const RefCountedString original = "copy me";
		const RefCountedString copy(original);

		CHECK(copy.Size() == original.Size());
		CHECK_THAT(copy.CString(), Equals(original.CString()));
		CHECK_THAT(copy.CString(), Equals("copy me"));
	}

	SECTION("Copy assigned string equals original")
	{
		const RefCountedString original = "assign me";
		RefCountedString copy;
		copy = original;

		CHECK_THAT(copy.CString(), Equals("assign me"));
	}

	SECTION("Copy is noexcept")
	{
		STATIC_CHECK(std::is_nothrow_copy_constructible_v<RefCountedString>);
		STATIC_CHECK(std::is_nothrow_copy_assignable_v<RefCountedString>);
	}

	SECTION("Copies share storage")
	{
		const RefCountedString a = "shared";
		const RefCountedString b(a);

		CHECK(a.CString() == b.CString());
	}

	SECTION("Copy outlives original")
	{
		RefCountedString copy;
		{
			const RefCountedString original = "outlive me";
			copy = original;
		}
		CHECK_THAT(copy.CString(), Equals("outlive me"));
		CHECK(copy.Size() == 10);
	}
}


TEST_CASE("RefCountedString - Move semantics")
{
	SECTION("Move constructed string takes original content")
	{
		RefCountedString original = "move me";
		const RefCountedString moved(std::move(original));

		CHECK_THAT(moved.CString(), Equals("move me"));
		CHECK(moved.Size() == 7);
	}

	SECTION("Move assigned string takes original content")
	{
		RefCountedString original = "move assign";
		RefCountedString moved;
		moved = std::move(original);

		CHECK_THAT(moved.CString(), Equals("move assign"));
	}

	SECTION("Move is noexcept")
	{
		STATIC_CHECK(std::is_nothrow_move_constructible_v<RefCountedString>);
		STATIC_CHECK(std::is_nothrow_move_assignable_v<RefCountedString>);
	}
}


TEST_CASE("RefCountedString - Stream output")
{
	SECTION("operator<< writes content")
	{
		const RefCountedString s = "stream me";
		std::ostringstream oss;
		oss << s;
		CHECK_THAT(oss.str(), Equals("stream me"));
	}

	SECTION("operator<< writes nothing for default-constructed")
	{
		const RefCountedString s;
		std::ostringstream oss;
		oss << s;
		CHECK(oss.str().empty());
	}
}


TEST_CASE("RefCountedString - Type traits")
{
	STATIC_CHECK(std::is_default_constructible_v<RefCountedString>);
	STATIC_CHECK(std::is_copy_constructible_v<RefCountedString>);
	STATIC_CHECK(std::is_move_constructible_v<RefCountedString>);
	STATIC_CHECK(std::is_copy_assignable_v<RefCountedString>);
	STATIC_CHECK(std::is_move_assignable_v<RefCountedString>);

	// The whole point of this type: cheap, throwing-free copies.
	STATIC_CHECK(std::is_nothrow_copy_constructible_v<RefCountedString>);
	STATIC_CHECK(std::is_nothrow_copy_assignable_v<RefCountedString>);
	STATIC_CHECK(std::is_nothrow_move_constructible_v<RefCountedString>);
	STATIC_CHECK(std::is_nothrow_move_assignable_v<RefCountedString>);
}
