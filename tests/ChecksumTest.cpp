#include <SecUtility/Misc/Checksum.hpp>
#include <SecUtility/Misc/Random.hpp>

#include <catch2/catch_test_macros.hpp>
#include <sstream>
#include <numeric>
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


//----------------------------------------------------------------------------------------------------------------------
// Checksum Algorithm Tests
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("SoftwareCrc32 - Known test vectors")
{
	SECTION("Empty data")
	{
		constexpr Checksum32 crc = SoftwareCrc32(nullptr, 0);
		STATIC_CHECK(std::to_underlying(crc) == 0);
	}

	SECTION("Single byte 0x00")
	{
		constexpr std::uint8_t data[] = {0x00};
		constexpr Checksum32 crc = SoftwareCrc32(data, 1);
		STATIC_CHECK(std::to_underlying(crc) == 0xD202EF8D);
	}

	SECTION("Single byte 0xFF")
	{
		constexpr std::uint8_t data[] = {0xFF};
		constexpr Checksum32 crc = SoftwareCrc32(data, 1);
		STATIC_CHECK(std::to_underlying(crc) == 0xFF000000);
	}

	SECTION("Two bytes 0x00 0x00")
	{
		constexpr std::uint8_t data[] = {0x00, 0x00};
		constexpr Checksum32 crc = SoftwareCrc32(data, 2);
		STATIC_CHECK(std::to_underlying(crc) == 0x41D912FF);
	}

	SECTION("Two bytes 0xFF 0xFF")
	{
		constexpr std::uint8_t data[] = {0xFF, 0xFF};
		constexpr Checksum32 crc = SoftwareCrc32(data, 2);
		STATIC_CHECK(std::to_underlying(crc) == 0xFFFF0000);
	}

	SECTION("String '123456789'")
	{
		constexpr std::uint8_t data[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};

		constexpr Checksum32 crc = SoftwareCrc32(data, 9);
		STATIC_CHECK(std::to_underlying(crc) == 0xCBF43926);
	}

	SECTION("String 'hello world'")
	{
		constexpr std::uint8_t data[] = "hello world";
		constexpr Checksum32 crc = SoftwareCrc32(data, 11);
		STATIC_CHECK(std::to_underlying(crc) == 0x0D4A1185);
	}

	SECTION("String 'The quick brown fox jumps over the lazy dog'")
	{
		constexpr std::uint8_t data[] = "The quick brown fox jumps over the lazy dog";
		constexpr Checksum32 crc = SoftwareCrc32(data, sizeof(data) - 1);
		STATIC_CHECK(std::to_underlying(crc) == 0x414FA339);
	}

	SECTION("String 'The quick brown fox jumps over the lazy dog.'")
	{
		constexpr std::uint8_t data[] = "The quick brown fox jumps over the lazy dog.";
		constexpr Checksum32 crc = SoftwareCrc32(data, sizeof(data) - 1);
		STATIC_CHECK(std::to_underlying(crc) == 0x519025E9);
	}

	SECTION("String 'abcdefghijklmnopqrstuvwxyz'")
	{
		constexpr std::uint8_t data[] = "abcdefghijklmnopqrstuvwxyz";
		constexpr Checksum32 crc = SoftwareCrc32(data, sizeof(data) - 1);
		STATIC_CHECK(std::to_underlying(crc) == 0x4C2750BD);
	}

	SECTION("String 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'")
	{
		constexpr std::uint8_t data[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
		constexpr Checksum32 crc = SoftwareCrc32(data, sizeof(data) - 1);
		STATIC_CHECK(std::to_underlying(crc) == 0x1FC2E6D2);
	}

	SECTION("String 'various CRC algorithms input data'")
	{
		constexpr std::uint8_t data[] = "various CRC algorithms input data";
		constexpr Checksum32 crc = SoftwareCrc32(data, sizeof(data) - 1);
		STATIC_CHECK(std::to_underlying(crc) == 0x9BD366AE);
	}
}


TEST_CASE("SoftwareCrc32 - Incremental computation")
{
	SECTION("Split computation matches whole")
	{
		constexpr std::uint8_t data[] = "123456789";

		// Compute all at once
		constexpr Checksum32 crc1 = SoftwareCrc32(data, 9);

		// Compute incrementally
		constexpr Checksum32 crc2 = SoftwareCrc32(data, 4);
		// constexpr Checksum32 crc3 = SoftwareCrc32(data + 4, 5, crc2);
		constexpr Checksum32 crc3 = SoftwareCrc32(data + 4, 5, Checksum32{crc2 ^ 0xFFFFFFFF});

		STATIC_CHECK(crc1 == crc3);
	}

	SECTION("Incremental with empty data")
	{
		constexpr std::uint8_t data[] = "hello";

		constexpr Checksum32 crc = SoftwareCrc32(data, 5);
		constexpr Checksum32 crc2 = SoftwareCrc32(data, 0, Checksum32{crc ^ 0xFFFFFFFF});  // Add empty data

		STATIC_CHECK(std::to_underlying(crc) == std::to_underlying(crc2));
	}
}

TEST_CASE("SoftwareCrc32C - Known test vectors")
{
	SECTION("Empty data")
	{
		constexpr Checksum32 s = SoftwareCrc32C(nullptr, 0);
		STATIC_CHECK(std::to_underlying(s) == 0);
	}

	SECTION("String '123456789'")
	{
		constexpr std::uint8_t data[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};

		constexpr Checksum32 s = SoftwareCrc32C(data, 9);
		STATIC_CHECK(std::to_underlying(s) == 0xe3069283);
	}

	SECTION("String 'The quick brown fox jumps over the lazy dog'")
	{
		constexpr std::uint8_t data[] = "The quick brown fox jumps over the lazy dog";

		constexpr Checksum32 s = SoftwareCrc32C(data, sizeof(data) - 1);
		STATIC_CHECK(std::to_underlying(s) == 0x22620404);
	}

	SECTION("String 'message digest'")
	{
		constexpr std::uint8_t data[] = "message digest";

		constexpr Checksum32 s = SoftwareCrc32C(data, sizeof(data) - 1);
		STATIC_CHECK(std::to_underlying(s) == 0x02bd79d0);
	}

	SECTION("String 'abcdefghijklmnopqrstuvwxyz'")
	{
		constexpr std::uint8_t data[] = "abcdefghijklmnopqrstuvwxyz";

		constexpr Checksum32 s = SoftwareCrc32C(data, sizeof(data) - 1);
		STATIC_CHECK(std::to_underlying(s) == 0x9ee6ef25);
	}

	SECTION("String 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'")
	{
		constexpr std::uint8_t data[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

		constexpr Checksum32 s = SoftwareCrc32C(data, sizeof(data) - 1);
		STATIC_CHECK(std::to_underlying(s) == 0xa245d57d);
	}

	SECTION("0x00 .. 0x1F")
	{
		std::uint8_t data[32] = {};
		std::iota(std::begin(data), std::end(data), std::uint8_t{0});

		const Checksum32 s = SoftwareCrc32C(data, 32);
		CHECK(std::to_underlying(s) == 0x46dd794e);
	}

	SECTION("0x1F .. 0x00")
	{
		std::uint8_t data[32] = {};
		std::iota(std::rbegin(data), std::rend(data), std::uint8_t{0});

		const Checksum32 s = SoftwareCrc32C(data, 32);
		CHECK(std::to_underlying(s) == 0x113fdb5c);
	}
}

TEST_CASE("HardwareCrc32C - Known test vectors")
{
#if SECUTILITY_HAS_HARDWARE_CRC32C
	SECTION("Empty data")
	{
		const Checksum32 h = HardwareCrc32C(nullptr, 0);
		CHECK(std::to_underlying(h) == 0);
	}

	SECTION("String '123456789'")
	{
		constexpr std::uint8_t data[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};

		const Checksum32 h = HardwareCrc32C(data, 9);
		CHECK(std::to_underlying(h) == 0xe3069283);
	}

	SECTION("String 'The quick brown fox jumps over the lazy dog'")
	{
		constexpr std::uint8_t data[] = "The quick brown fox jumps over the lazy dog";

		const Checksum32 h = HardwareCrc32C(data, sizeof(data) - 1);
		CHECK(std::to_underlying(h) == 0x22620404);
	}

	SECTION("String 'message digest'")
	{
		constexpr std::uint8_t data[] = "message digest";

		const Checksum32 h = HardwareCrc32C(data, sizeof(data) - 1);
		CHECK(std::to_underlying(h) == 0x02bd79d0);
	}

	SECTION("String 'abcdefghijklmnopqrstuvwxyz'")
	{
		constexpr std::uint8_t data[] = "abcdefghijklmnopqrstuvwxyz";

		const Checksum32 h = HardwareCrc32C(data, sizeof(data) - 1);
		CHECK(std::to_underlying(h) == 0x9ee6ef25);
	}

	SECTION("String 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'")
	{
		constexpr std::uint8_t data[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

		const Checksum32 h = HardwareCrc32C(data, sizeof(data) - 1);
		CHECK(std::to_underlying(h) == 0xa245d57d);
	}

	SECTION("0x00 .. 0x1F")
	{
		std::uint8_t data[32] = {};
		std::iota(std::begin(data), std::end(data), std::uint8_t{0});

		const Checksum32 h = HardwareCrc32C(data, 32);
		CHECK(std::to_underlying(h) == 0x46dd794e);
	}

	SECTION("0x1F .. 0x00")
	{
		std::uint8_t data[32] = {};
		std::iota(std::rbegin(data), std::rend(data), std::uint8_t{0});

		const Checksum32 h = HardwareCrc32C(data, 32);
		CHECK(std::to_underlying(h) == 0x113fdb5c);
	}
#else
	SKIP("Test requires SSE4.2 hardware. One may also need to pass -march=... to compiler");
#endif
}


TEST_CASE("SoftwareCrc32CC - Incremental computation")
{
	SECTION("Split computation matches whole")
	{
		constexpr std::uint8_t data[] = "123456789";

		// Compute all at once
		constexpr Checksum32 crc1 = SoftwareCrc32C(data, 9);

		// Compute incrementally
		constexpr Checksum32 crc2 = SoftwareCrc32C(data, 4);
		// constexpr Checksum32 crc3 = SoftwareCrc32C(data + 4, 5, crc2);
		constexpr Checksum32 crc3 = SoftwareCrc32C(data + 4, 5, Checksum32{crc2 ^ 0xFFFFFFFF});

		STATIC_CHECK(crc1 == crc3);
	}

	SECTION("Incremental with empty data")
	{
		constexpr std::uint8_t data[] = "hello";

		constexpr Checksum32 crc = SoftwareCrc32C(data, 5);
		constexpr Checksum32 crc2 = SoftwareCrc32C(data, 0, Checksum32{crc ^ 0xFFFFFFFF});  // Add empty data

		STATIC_CHECK(std::to_underlying(crc) == std::to_underlying(crc2));
	}
}


TEST_CASE("Slicing-by-8 Crc32")
{
	std::vector<std::int32_t> data(257);
	for (auto& d : data)
	{
		d = SecUtility::Random::NextInt32();
	}

	{
		const auto s = SoftwareCrc32(reinterpret_cast<const std::uint8_t*>(data.data()) + 1,
		                             data.size() * sizeof(std::int32_t) - 1);
		const auto s8 = SlicedSoftwareCrc32<8>(reinterpret_cast<const std::uint8_t*>(data.data()) + 1,
		                                       data.size() * sizeof(std::int32_t) - 1);
		CHECK(s == s8);
	}
	{
		const auto s = SoftwareCrc32C(reinterpret_cast<const std::uint8_t*>(data.data()) + 1,
		                              data.size() * sizeof(std::int32_t) - 1);
		const auto s8 = SlicedSoftwareCrc32C<8>(reinterpret_cast<const std::uint8_t*>(data.data()) + 1,
		                                        data.size() * sizeof(std::int32_t) - 1);
		CHECK(s == s8);
	}

	constexpr std::uint8_t verse[] =
	        "Love is patient, love is kind. It does not envy, it does not boast, it is not proud. It is not rude, it "
	        "is not self-seeking, it is not easily angered, it keeps no record of wrongs.";

	{
		constexpr auto s = SoftwareCrc32(verse, sizeof(verse) - 1);
		constexpr auto s8 = SlicedSoftwareCrc32<8>(verse, sizeof(verse) - 1);
		STATIC_CHECK(s == s8);
	}
	{
		constexpr auto s = SoftwareCrc32C(verse, sizeof(verse) - 1);
		constexpr auto s8 = SlicedSoftwareCrc32C<8>(verse, sizeof(verse) - 1);
		STATIC_CHECK(s == s8);
	}
}


TEST_CASE("Slicing-by-16 Crc32")
{
	std::vector<std::int32_t> data(257);
	for (auto& d : data)
	{
		d = SecUtility::Random::NextInt32();
	}

	{
		const auto s = SoftwareCrc32(reinterpret_cast<const std::uint8_t*>(data.data()) + 1,
		                             data.size() * sizeof(std::int32_t) - 1);
		const auto s8 = SlicedSoftwareCrc32<16>(reinterpret_cast<const std::uint8_t*>(data.data()) + 1,
		                                        data.size() * sizeof(std::int32_t) - 1);
		CHECK(s == s8);
	}
	{
		const auto s = SoftwareCrc32C(reinterpret_cast<const std::uint8_t*>(data.data()) + 1,
		                              data.size() * sizeof(std::int32_t) - 1);
		const auto s8 = SlicedSoftwareCrc32C<16>(reinterpret_cast<const std::uint8_t*>(data.data()) + 1,
		                                         data.size() * sizeof(std::int32_t) - 1);
		CHECK(s == s8);
	}

	constexpr std::uint8_t verse[] =
		"Love is patient, love is kind. It does not envy, it does not boast, it is not proud. It is not rude, it "
		"is not self-seeking, it is not easily angered, it keeps no record of wrongs.";

	{
		const auto s = SoftwareCrc32(verse, sizeof(verse) - 1);
		const auto s8 = SlicedSoftwareCrc32<16>(verse, sizeof(verse) - 1);
		CHECK(s == s8);
	}
	{
		const auto s = SoftwareCrc32C(verse, sizeof(verse) - 1);
		const auto s8 = SlicedSoftwareCrc32C<16>(verse, sizeof(verse) - 1);
		CHECK(s == s8);
	}
}
