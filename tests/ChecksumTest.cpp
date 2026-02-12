#include <SecUtility/Misc/Checksum.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <type_traits>

using namespace SecUtility::Checksum;


TEST_CASE("Checksum32 - Enum class properties")
{
	SECTION("Is an enum class")
	{
		STATIC_CHECK(std::is_enum_v<Checksum32>);
	}

	SECTION("Underlying type is uint32_t")
	{
		STATIC_CHECK(std::is_same_v<std::underlying_type_t<Checksum32>, std::uint32_t>);
	}

	SECTION("Can be constructed from uint32_t")
	{
		constexpr Checksum32 c1{0x12345678};
		STATIC_CHECK(std::to_underlying(c1) == 0x12345678);
	}

	SECTION("Can be constructed with braced initialization")
	{
		constexpr Checksum32 c{0xFFFFFFFF};
		STATIC_CHECK(std::to_underlying(c) == 0xFFFFFFFF);
	}
}


TEST_CASE("Checksum64 - Enum class properties")
{
	SECTION("Is an enum class")
	{
		STATIC_CHECK(std::is_enum_v<Checksum64>);
	}

	SECTION("Underlying type is uint64_t")
	{
		STATIC_CHECK(std::is_same_v<std::underlying_type_t<Checksum64>, std::uint64_t>);
	}

	SECTION("Can be constructed from uint64_t")
	{
		constexpr Checksum64 c1{0x123456789ABCDEF0ULL};
		STATIC_CHECK(std::to_underlying(c1) == 0x123456789ABCDEF0ULL);
	}

	SECTION("Can be constructed with braced initialization")
	{
		constexpr Checksum64 c{0xFFFFFFFFFFFFFFFFULL};
		STATIC_CHECK(std::to_underlying(c) == 0xFFFFFFFFFFFFFFFFULL);
	}
}


TEST_CASE("Checksum32 - Bitwise XOR operator")
{
	SECTION("Checksum32 ^ integral type")
	{
		constexpr Checksum32 c{0xF0F0F0F0};
		constexpr std::uint32_t result = c ^ 0xAAAAAAAA;

		STATIC_CHECK(result == 0x5A5A5A5A);
	}

	SECTION("Integral type ^ Checksum32")
	{
		constexpr Checksum32 c{0xF0F0F0F0};
		constexpr std::uint32_t result = 0xAAAAAAAA ^ c;

		STATIC_CHECK(result == 0x5A5A5A5A);
	}

	SECTION("Checksum32 ^ Checksum32 (via underlying)")
	{
		constexpr Checksum32 c1{0x12345678};
		constexpr Checksum32 c2{0x87654321};
		constexpr auto result = std::to_underlying(c1) ^ std::to_underlying(c2);

		STATIC_CHECK(result == 0x95511559);
	}

	SECTION("XOR with zero")
	{
		constexpr Checksum32 c{0xABCDEF00};
		constexpr auto result = c ^ 0;

		STATIC_CHECK(result == 0xABCDEF00);
	}

	SECTION("XOR with itself yields zero")
	{
		constexpr Checksum32 c{0x12345678};
		constexpr auto result = c ^ std::to_underlying(c);

		STATIC_CHECK(result == 0);
	}
}


TEST_CASE("Checksum64 - Bitwise XOR operator")
{
	SECTION("Checksum64 ^ integral type")
	{
		constexpr Checksum64 c{0xF0F0F0F0F0F0F0F0ULL};
		constexpr std::uint64_t result = c ^ 0x0F0F0F0F0F0F0F0FULL;

		STATIC_CHECK(result == 0xFFFFFFFFFFFFFFFFULL);
	}

	SECTION("Integral type ^ Checksum64")
	{
		constexpr Checksum64 c{0xAAAAAAAAAAAAAAAAULL};
		constexpr std::uint64_t result = 0x5555555555555555ULL ^ c;

		STATIC_CHECK(result == 0xFFFFFFFFFFFFFFFFULL);
	}

	SECTION("XOR with zero")
	{
		constexpr Checksum64 c{0x123456789ABCDEF0ULL};
		constexpr auto result = c ^ 0ULL;

		STATIC_CHECK(result == 0x123456789ABCDEF0ULL);
	}
}


TEST_CASE("Checksum32 - Bitwise shift operators")
{
	SECTION("Left shift")
	{
		constexpr Checksum32 c{0x00000001};
		constexpr auto result = c << 4;

		STATIC_CHECK(result == 0x00000010);
	}

	SECTION("Right shift")
	{
		constexpr Checksum32 c{0x10000000};
		constexpr auto result = c >> 4;

		STATIC_CHECK(result == 0x01000000);
	}

	SECTION("Left shift by larger amounts")
	{
		constexpr  Checksum32 c{1};
		STATIC_CHECK((c << 8) == 0x00000100);
		STATIC_CHECK((c << 16) == 0x00010000);
		STATIC_CHECK((c << 24) == 0x01000000);
	}

	SECTION("Right shift by larger amounts")
	{
		constexpr Checksum32 c{0xFF000000};
		STATIC_CHECK((c >> 8) == 0x00FF0000);
		STATIC_CHECK((c >> 16) == 0x0000FF00);
		STATIC_CHECK((c >> 24) == 0x000000FF);
	}

	SECTION("Shift by zero")
	{
		constexpr Checksum32 c{0x12345678};
		STATIC_CHECK((c << 0) == 0x12345678);
		STATIC_CHECK((c >> 0) == 0x12345678);
	}
}


TEST_CASE("Checksum64 - Bitwise shift operators")
{
	SECTION("Left shift")
	{
		constexpr Checksum64 c{0x0000000000000001ULL};
		constexpr auto result = c << 8;

		STATIC_CHECK(result == 0x0000000000000100ULL);
	}

	SECTION("Right shift")
	{
		constexpr Checksum64 c{0x0100000000000000ULL};
		constexpr auto result = c >> 8;

		STATIC_CHECK(result == 0x0001000000000000ULL);
	}

	SECTION("Large shifts")
	{
		constexpr Checksum64 c{1};
		STATIC_CHECK((c << 32) == 0x0000000100000000ULL);
		STATIC_CHECK((c >> 32) == 0x0000000000000000ULL);
	}
}


TEST_CASE("Checksum32 - Bitwise NOT operator")
{
	SECTION("NOT operation")
	{
		constexpr Checksum32 c{0x12345678};
		constexpr auto result = ~c;

		// ~0x12345678 = 0xEDCBA987
		STATIC_CHECK(result == 0xEDCBA987);
	}

	SECTION("NOT twice returns original")
	{
		constexpr Checksum32 c{0xABCDEF00};
		constexpr auto result = ~~c;

		STATIC_CHECK(result == 0xABCDEF00);
	}

	SECTION("NOT of zero is all ones")
	{
		constexpr Checksum32 c{0};
		constexpr auto result = ~c;

		STATIC_CHECK(result == 0xFFFFFFFF);
	}

	SECTION("NOT of all ones is zero")
	{
		constexpr Checksum32 c{0xFFFFFFFF};
		constexpr auto result = ~c;

		STATIC_CHECK(result == 0);
	}
}


TEST_CASE("Checksum64 - Bitwise NOT operator")
{
	SECTION("NOT operation")
	{
		constexpr Checksum64 c{0x123456789ABCDEF0ULL};
		constexpr auto result = ~c;

		// ~0x123456789ABCDEF0 = 0xEDCBA9876543210F
		STATIC_CHECK(result == 0xEDCBA9876543210FULL);
	}

	SECTION("NOT of zero is all ones")
	{
		constexpr Checksum64 c{0};
		constexpr auto result = ~c;

		STATIC_CHECK(result == 0xFFFFFFFFFFFFFFFFULL);
	}

	SECTION("NOT of all ones is zero")
	{
		constexpr Checksum64 c{0xFFFFFFFFFFFFFFFFULL};
		constexpr auto result = ~c;

		STATIC_CHECK(result == 0);
	}
}


TEST_CASE("Checksum32 - Stream output operator")
{
	SECTION("Formats as hexadecimal with prefix")
	{
		Checksum32 c{0x12345678};
		std::stringstream ss;
		ss << c;

		CHECK(ss.str() == "0x12345678");
	}

	SECTION("Pads with zeros to 8 characters")
	{
		Checksum32 c{0xABC};
		std::stringstream ss;
		ss << c;

		CHECK(ss.str() == "0x00000ABC");
	}

	SECTION("Uses uppercase hex digits")
	{
		Checksum32 c{0xFFFFFFFF};
		std::stringstream ss;
		ss << c;

		CHECK(ss.str() == "0xFFFFFFFF");
	}

	SECTION("Preserves stream state")
	{
		Checksum32 c{0xDEADBEEF};

		std::stringstream ss;
		ss << std::dec << 42 << " ";  // Set to decimal
		ss << c;
		ss << " " << 99;  // Should still be decimal

		CHECK(ss.str() == "42 0xDEADBEEF 99");
	}

	SECTION("Maximum value")
	{
		Checksum32 c{0xFFFFFFFF};
		std::stringstream ss;
		ss << c;

		CHECK(ss.str() == "0xFFFFFFFF");
	}

	SECTION("Zero value")
	{
		Checksum32 c{0};
		std::stringstream ss;
		ss << c;

		CHECK(ss.str() == "0x00000000");
	}
}


TEST_CASE("Checksum64 - Stream output operator")
{
	SECTION("Formats as hexadecimal with prefix")
	{
		Checksum64 c{0x123456789ABCDEF0ULL};
		std::stringstream ss;
		ss << c;

		CHECK(ss.str() == "0x123456789ABCDEF0");
	}

	SECTION("Pads with zeros to 16 characters")
	{
		Checksum64 c{0xABC};
		std::stringstream ss;
		ss << c;

		CHECK(ss.str() == "0x0000000000000ABC");
	}

	SECTION("Uses uppercase hex digits")
	{
		Checksum64 c{0xFFFFFFFFFFFFFFFFULL};
		std::stringstream ss;
		ss << c;

		CHECK(ss.str() == "0xFFFFFFFFFFFFFFFF");
	}

	SECTION("Maximum value")
	{
		Checksum64 c{0xFFFFFFFFFFFFFFFFULL};
		std::stringstream ss;
		ss << c;

		CHECK(ss.str() == "0xFFFFFFFFFFFFFFFF");
	}

	SECTION("Zero value")
	{
		Checksum64 c{0};
		std::stringstream ss;
		ss << c;

		CHECK(ss.str() == "0x0000000000000000");
	}
}


TEST_CASE("Checksum32 - Combined operations")
{
	SECTION("Shift then XOR")
	{
		constexpr Checksum32 c{0x00000001};
		constexpr auto shifted = c << 8;  // 0x00000100
		constexpr auto xored = shifted ^ 0x00000101;

		CHECK(xored == 0x00000001);
	}

	SECTION("XOR then shift")
	{
		constexpr Checksum32 c{0xF0F0F0F0};

		constexpr auto xored = c ^ 0xFFFFFFFF;
		CHECK(xored == 0xF0F0F0F);

		constexpr auto shifted = xored >> 4;
		CHECK(shifted == 0xF0F0F0);
	}

	SECTION("NOT then XOR")
	{
		constexpr Checksum32 c{0x12345678};

		constexpr auto noted = ~c;  // 0xEDCBA987
		STATIC_CHECK(noted == 0xEDCBA987);

		constexpr auto xored = noted ^ 0xFFFFFFFF;
		STATIC_CHECK(xored == 0x12345678);
	}
}


TEST_CASE("Checksum32 - Type safety")
{
	SECTION("Cannot implicitly convert to uint32_t")
	{
		constexpr Checksum32 c{0x12345678};

		// This should NOT compile:
		// std::uint32_t value = c;

		// But this should:
		constexpr std::uint32_t value = std::to_underlying(c);
		CHECK(value == 0x12345678);
	}

	SECTION("Cannot mix Checksum32 and Checksum64 without explicit conversion")
	{
		STATIC_CHECK(std::is_assignable_v<Checksum32&, Checksum32>);
		STATIC_CHECK(std::is_assignable_v<Checksum64&, Checksum64>);
		STATIC_CHECK(!std::is_assignable_v<Checksum32&, Checksum64>);
		STATIC_CHECK(!std::is_assignable_v<Checksum64&, Checksum32>);
	}
}


TEST_CASE("Checksum types - Common values")
{
	SECTION("Default CRC initial value (0xFFFFFFFF)")
	{
		constexpr Checksum32 defaultCrc32{0xFFFFFFFF};
		CHECK(std::to_underlying(defaultCrc32) == 0xFFFFFFFF);

		constexpr Checksum64 defaultCrc64{0xFFFFFFFFFFFFFFFFULL};
		CHECK(std::to_underlying(defaultCrc64) == 0xFFFFFFFFFFFFFFFFULL);
	}

	SECTION("Zero checksum")
	{
		constexpr Checksum32 zero32{0};
		constexpr Checksum64 zero64{0};

		CHECK(std::to_underlying(zero32) == 0);
		CHECK(std::to_underlying(zero64) == 0);
	}
}
