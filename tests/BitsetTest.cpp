//
// Created by Andy on 5/1/2026.
//

// --- SEC_ASSERT interception -------------------------------------------------
// bitset.hpp uses SEC_ASSERT(cond) for all precondition checks.  If SEC_ASSERT
// is defined before the include, that definition is used throughout – including
// in every template instantiation triggered by this translation unit.
// We define it to throw std::logic_error so that REQUIRE_THROWS_AS can catch
// precondition violations without aborting the process.
// ---------------------------------------------------------------------------
#include <stdexcept>  // std::logic_error

#define SEC_ASSERT(cond)                                                                                               \
	do                                                                                                                 \
	{                                                                                                                  \
		if (!(cond))                                                                                                   \
			throw std::logic_error("SEC_ASSERT failed: " #cond);                                                       \
	} while (0)
#define SEC_NOEXCEPT noexcept(false)

#include <SecUtility/Collection/Bitset.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

// =============================================================================
//  Helpers
// =============================================================================
static std::string RandomBitString(const std::size_t len, const std::uint64_t seed)
{
	std::mt19937_64 rng(seed);
	std::string s(len, '0');
	for (char& c : s)
	{
		c = (rng() & 1u) ? '1' : '0';
	}
	return s;
}

// static std::size_t CountOnesInString(const std::string& s)
// {
// 	return static_cast<std::size_t>(std::count(s.begin(), s.end(), '1'));
// }

using namespace SecUtility;


// =============================================================================
//  Helper to verify bitset matches string pattern
// =============================================================================
template <typename BitsetType>
static void VerifyBitset(const BitsetType& bs, const std::string& expected)
{
	REQUIRE(bs.Size() == expected.size());
	for (std::size_t i = 0; i < bs.Size(); ++i)
	{
		const bool actualBit = bs[i];
		const bool expectedBit = expected[i] != '0';
		if (actualBit != expectedBit)
		{
			FAIL_CHECK("Bit mismatch at index " << i << ": expected " << expectedBit << ", got " << actualBit);
		}
	}
}


// =============================================================================
//  Test Construction
// =============================================================================
TEST_CASE("Bitset construction", "[bitset][construction]")
{
	SECTION("Default construction - Bitset<N>")
	{
		{
			Bitset<0> bs;
			STATIC_REQUIRE(bs.Size() == 0);
		}{
			Bitset<64> bs;
			STATIC_REQUIRE(bs.Size() == 64);
		}{
			Bitset<32> bs;
			STATIC_REQUIRE(bs.Size() == 32);
		}{
			Bitset<1024> bs;
			STATIC_REQUIRE(bs.Size() == 1024);
		}{
			Bitset<264> bs;
			STATIC_REQUIRE(bs.Size() == 264);
		}{
			Bitset<164> bs;
			STATIC_REQUIRE(bs.Size() == 164);
		}
	}

	SECTION("Default construction - DynamicBitset")
	{
		DynamicBitset bs;
		REQUIRE(bs.Size() == 0);
	}

	SECTION("DynamicBitset size construction - all false")
	{
		std::size_t size = GENERATE(1, 7, 8, 63, 64, 65, 127, 128, 129, 256);
		{
			DynamicBitset bs(size, false);
			REQUIRE(bs.Size() == size);
		}
		{
			DynamicBitset bs(size, true);
			REQUIRE(bs.Size() == size);
		}
	}
}


// =============================================================================
//  Test Element Access
// =============================================================================
TEST_CASE("Element access", "[bitset][access]")
{
	SECTION("operator[] - read/write bits")
	{
		const std::size_t size = GENERATE(8, 63, 64, 65, 128);
		const auto seed = GENERATE(42, 69, 123, 999);

		const std::string pattern = RandomBitString(size, seed);
		DynamicBitset bs(size);

		for (std::size_t i = 0; i < size; ++i)
		{
			CHECK(bs.IsOne(i) == static_cast<bool>(bs[i]));
			CHECK(bs.IsZero(i) == !static_cast<bool>(bs[i]));

			bs[i] = pattern[i] != '0';
			CHECK(std::as_const(bs)[i] == static_cast<bool>(bs[i]));  // lhs returns bool, rhs returns proxy
			CHECK(std::as_const(bs)[i] == (pattern[i] != '0'));

			{
				const bool old = bs[i];
				bs[i].Flip();
				CHECK(std::as_const(bs)[i] == !old);
			}
		}

		REQUIRE_THROWS_AS(bs[size + 1], std::logic_error);
		REQUIRE_THROWS_AS(bs[size + 2], std::logic_error);
		REQUIRE_THROWS_AS(bs[std::numeric_limits<std::size_t>::max()], std::logic_error);
	}
}


// // =============================================================================
// //  Test Modification Operations
// // =============================================================================
TEST_CASE("Modification operations", "[bitset][modify]")
{
	SECTION("Set and Reset")
	{
		std::size_t size = GENERATE(8, 64, 65, 128);
		auto seed = GENERATE(42, 123);

		std::string pattern = RandomBitString(size, seed);
		DynamicBitset bs(size);

		for (std::size_t i = 0; i < size; ++i)
		{
			if (pattern[i] == '1')
			{
				bs.Set(i);
			}
		}

		VerifyBitset(bs, pattern);

		// Reset all bits
		for (std::size_t i = 0; i < size; ++i)
		{
			bs.Reset(i);
			REQUIRE(bs[i] == false);
		}
	}

	SECTION("Set with value parameter")
	{
		Bitset<128> bs;

		for (std::size_t i = 0; i < 128; i += 2)
		{
			bs.Set(i, true);
			bs.Set(i + 1, false);
		}

		for (std::size_t i = 0; i < 128; ++i)
		{
			REQUIRE(bs[i] == (i % 2 == 0));
		}
	}

	SECTION("Flip")
	{
		std::size_t size = GENERATE(8, 64, 65, 128);
		auto seed = GENERATE(42, 123);

		std::string original = RandomBitString(size, seed);
		std::string flipped;
		flipped.reserve(size);

		for (char c : original)
		{
			flipped += c != '0' ? '0' : '1';
		}

		DynamicBitset bs(size);
		for (std::size_t i = 0; i < size; ++i)
		{
			bs.Set(i, original[i] != '0');
		}

		// Flip every bit
		for (std::size_t i = 0; i < size; ++i)
		{
			bs.Flip(i);
		}

		VerifyBitset(bs, flipped);
	}

	SECTION("Method chaining")
	{
		Bitset<128> bs;

		bs.Set(0).Set(1).Set(2);
		REQUIRE(bs[0]);
		REQUIRE(bs[1]);
		REQUIRE(bs[2]);

		bs.Reset(0).Reset(1);
		REQUIRE(!bs[0]);
		REQUIRE(!bs[1]);
		REQUIRE(bs[2]);

		bs.Flip(2).Flip(3);
		REQUIRE(!bs[2]);
		REQUIRE(bs[3]);
	}
}


// // =============================================================================
// //  Test BitReference Proxy
// // =============================================================================
// TEST_CASE("BitReference proxy", "[bitset][proxy]")
// {
// 	SECTION("Assignment from bool")
// 	{
// 		DynamicBitset bs(128);
// 		auto ref0 = bs[0];
// 		auto ref64 = bs[64];
//
// 		ref0 = true;
// 		ref64 = false;
//
// 		REQUIRE(bs[0] == true);
// 		REQUIRE(bs[64] == false);
//
// 		ref0 = false;
// 		ref64 = true;
//
// 		REQUIRE(bs[0] == false);
// 		REQUIRE(bs[64] == true);
// 	}
//
// 	SECTION("Assignment from another BitReference")
// 	{
// 		Bitset<128> bs1;
// 		DynamicBitset bs2(128);
//
// 		bs1.Set(0);
// 		bs2.Set(10, true);
//
// 		bs1[10] = bs2[0];
// 		bs2[20] = bs1[10];
//
// 		REQUIRE(bs1[10] == true);
// 		REQUIRE(bs2[20] == true);
// 	}
//
// 	SECTION("Cross-type BitReference assignment")
// 	{
// 		Bitset<128> fixed;
// 		DynamicBitset dynamic(128);
//
// 		fixed.Set(5, true);
// 		dynamic[10] = fixed[5];
//
// 		REQUIRE(dynamic[10] == true);
//
// 		dynamic.Set(15, false);
// 		fixed[20] = dynamic[15];
//
// 		REQUIRE(fixed[20] == false);
// 	}
//
// 	SECTION("BitReference Flip")
// 	{
// 		DynamicBitset bs(128);
//
// 		bs.Set(0, true);
// 		bs.Set(1, false);
//
// 		bs[0].Flip();
// 		bs[1].Flip();
//
// 		REQUIRE(bs[0] == false);
// 		REQUIRE(bs[1] == true);
// 	}
//
// 	SECTION("BitReference conversion to bool")
// 	{
// 		Bitset<64> bs;
// 		bs.Set(0);
// 		bs.Set(63);
//
// 		bool b0 = static_cast<bool>(bs[0]);
// 		bool b1 = static_cast<bool>(bs[1]);
// 		bool b63 = static_cast<bool>(bs[63]);
//
// 		REQUIRE(b0 == true);
// 		REQUIRE(b1 == false);
// 		REQUIRE(b63 == true);
// 	}
// }
//
//
// // =============================================================================
// //  Test Comparison
// // =============================================================================
// TEST_CASE("Comparison operators", "[bitset][compare]")
// {
// 	SECTION("Same type equality")
// 	{
// 		auto seed = GENERATE(42, 123, 999);
// 		std::string pattern = RandomBitString(128, seed);
//
// 		Bitset<128> bs1;
// 		Bitset<128> bs2;
//
// 		for (std::size_t i = 0; i < 128; ++i)
// 		{
// 			if (pattern[i] == '1')
// 			{
// 				bs1.Set(i);
// 				bs2.Set(i);
// 			}
// 		}
//
// 		REQUIRE(bs1 == bs2);
// 		REQUIRE(!(bs1 != bs2));
//
// 		bs2.Flip(0);
// 		REQUIRE(bs1 != bs2);
// 		REQUIRE(!(bs1 == bs2));
// 	}
//
// 	SECTION("Cross-type equality - same size")
// 	{
// 		auto seed = GENERATE(42, 123, 999);
// 		std::string pattern = RandomBitString(100, seed);
//
// 		Bitset<100> fixed;
// 		DynamicBitset dynamic(100);
//
// 		for (std::size_t i = 0; i < 100; ++i)
// 		{
// 			fixed.Set(i, pattern[i] == '1');
// 			dynamic.Set(i, pattern[i] == '1');
// 		}
//
// 		REQUIRE(fixed == dynamic);
// 		REQUIRE(dynamic == fixed);
// 	}
//
// 	SECTION("Cross-type equality - different sizes")
// 	{
// 		Bitset<128> bs1;
// 		DynamicBitset bs2(100);
//
// 		REQUIRE(bs1 != bs2);
// 		REQUIRE(!(bs1 == bs2));
// 	}
//
// 	SECTION("Empty bitsets are equal")
// 	{
// 		Bitset<0> bs1;
// 		DynamicBitset bs2;
// 		DynamicBitset bs3(0);
//
// 		REQUIRE(bs1 == bs2);
// 		REQUIRE(bs2 == bs3);
// 	}
// }
//
//
// // =============================================================================
// //  Test Copy and Move
// // =============================================================================
// TEST_CASE("Copy and move operations", "[bitset][copy][move]")
// {
// 	SECTION("Copy construction")
// 	{
// 		auto seed = GENERATE(42, 123);
// 		std::string pattern = RandomBitString(128, seed);
//
// 		Bitset<128> original;
// 		for (std::size_t i = 0; i < 128; ++i)
// 		{
// 			original.Set(i, pattern[i] == '1');
// 		}
//
// 		Bitset<128> copy(original);
//
// 		REQUIRE(copy == original);
// 		VerifyBitset(copy, pattern);
// 	}
//
// 	SECTION("Copy assignment")
// 	{
// 		auto seed = GENERATE(42, 123);
// 		std::string pattern = RandomBitString(100, seed);
//
// 		DynamicBitset original(100);
// 		for (std::size_t i = 0; i < 100; ++i)
// 		{
// 			original.Set(i, pattern[i] == '1');
// 		}
//
// 		DynamicBitset copy;
// 		copy = original;
//
// 		REQUIRE(copy == original);
// 		VerifyBitset(copy, pattern);
// 	}
//
// 	SECTION("Cross-type copy construction")
// 	{
// 		auto seed = 42;
// 		std::string pattern = RandomBitString(64, seed);
//
// 		DynamicBitset original(64);
// 		for (std::size_t i = 0; i < 64; ++i)
// 		{
// 			original.Set(i, pattern[i] == '1');
// 		}
//
// 		Bitset<64> copy(original);
// 		REQUIRE(copy == original);
// 		VerifyBitset(copy, pattern);
// 	}
//
// 	SECTION("Cross-type copy assignment")
// 	{
// 		auto seed = 42;
// 		std::string pattern = RandomBitString(80, seed);
//
// 		Bitset<80> original;
// 		for (std::size_t i = 0; i < 80; ++i)
// 		{
// 			original.Set(i, pattern[i] == '1');
// 		}
//
// 		DynamicBitset copy(80);
// 		copy = original;
//
// 		REQUIRE(copy == original);
// 		VerifyBitset(copy, pattern);
// 	}
//
// 	SECTION("Move construction")
// 	{
// 		DynamicBitset original(128, true);
//
// 		DynamicBitset movedTo(std::move(original));
//
// 		for (std::size_t i = 0; i < 128; ++i)
// 		{
// 			REQUIRE(movedTo[i] == true);
// 		}
// 	}
//
// 	SECTION("Move assignment")
// 	{
// 		DynamicBitset original(100, true);
//
// 		DynamicBitset movedTo{};
// 		movedTo = std::move(original);
//
// 		for (std::size_t i = 0; i < 100; ++i)
// 		{
// 			REQUIRE(movedTo[i] == true);
// 		}
// 	}
// }
//
//
// // =============================================================================
// //  Test Sanitization
// // =============================================================================
// TEST_CASE("Sanitization of unused bits", "[bitset][sanitize]")
// {
// 	SECTION("DynamicBitset constructor with true")
// 	{
// 		std::size_t size = GENERATE(1, 7, 8, 63, 64, 65, 127, 128, 129, 200);
//
// 		DynamicBitset bs(size, true);
//
// 		// All valid bits should be set
// 		for (std::size_t i = 0; i < size; ++i)
// 		{
// 			REQUIRE(bs[i] == true);
// 		}
// 	}
//
// 	SECTION("Cross-type construction preserves sanitization")
// 	{
// 		std::size_t size = GENERATE(65, 129, 200);
//
// 		DynamicBitset original(size, true);
// 		Bitset<256> copy(original);
//
// 		// Verify copy matches original in valid range
// 		for (std::size_t i = 0; i < size; ++i)
// 		{
// 			REQUIRE(copy[i] == original[i]);
// 		}
//
// 		// Verify unused bits are zero
// 		for (std::size_t i = size; i < 256; ++i)
// 		{
// 			REQUIRE(copy[i] == false);
// 		}
// 	}
//
// 	SECTION("DynamicBitset cross-type assignment")
// 	{
// 		DynamicBitset original(65, true);
// 		Bitset<128> copy;
// 		copy = original;
//
// 		// Verify copy matches original in valid range
// 		for (std::size_t i = 0; i < 65; ++i)
// 		{
// 			REQUIRE(copy[i] == original[i]);
// 		}
//
// 		// Verify unused bits are zero
// 		for (std::size_t i = 65; i < 128; ++i)
// 		{
// 			REQUIRE(copy[i] == false);
// 		}
// 	}
// }
//
//
// // =============================================================================
// //  Test Edge Cases
// // =============================================================================
// TEST_CASE("Edge cases", "[bitset][edge]")
// {
// 	SECTION("Single bit bitset")
// 	{
// 		Bitset<1> bs;
// 		REQUIRE(bs.Size() == 1);
// 		REQUIRE(!bs[0]);
//
// 		bs.Set(0);
// 		REQUIRE(bs[0]);
//
// 		bs.Flip(0);
// 		REQUIRE(!bs[0]);
// 	}
//
// 	SECTION("Block boundary sizes")
// 	{
// 		std::size_t size = GENERATE(63, 64, 65, 127, 128, 129);
//
// 		auto seed = 42;
// 		std::string pattern = RandomBitString(size, seed);
//
// 		DynamicBitset bs(size);
// 		for (std::size_t i = 0; i < size; ++i)
// 		{
// 			bs.Set(i, pattern[i] == '1');
// 		}
//
// 		VerifyBitset(bs, pattern);
// 	}
//
// 	SECTION("All bits set")
// 	{
// 		Bitset<128> bs;
//
// 		for (std::size_t i = 0; i < 128; ++i)
// 		{
// 			bs.Set(i);
// 		}
//
// 		for (std::size_t i = 0; i < 128; ++i)
// 		{
// 			REQUIRE(bs[i] == true);
// 		}
// 	}
//
// 	SECTION("All bits zero")
// 	{
// 		DynamicBitset bs(200);
//
// 		for (std::size_t i = 0; i < 200; ++i)
// 		{
// 			REQUIRE(bs[i] == false);
// 		}
//
// 		// Even after setting and resetting
// 		for (std::size_t i = 0; i < 200; ++i)
// 		{
// 			bs.Set(i);
// 			bs.Reset(i);
// 			REQUIRE(bs[i] == false);
// 		}
// 	}
//
// 	SECTION("Alternating pattern")
// 	{
// 		Bitset<256> bs;
//
// 		for (std::size_t i = 0; i < 256; ++i)
// 		{
// 			bs.Set(i, i % 2 == 0);
// 		}
//
// 		for (std::size_t i = 0; i < 256; ++i)
// 		{
// 			REQUIRE(bs[i] == (i % 2 == 0));
// 		}
// 	}
//
// 	SECTION("Large bitset")
// 	{
// 		std::size_t size = GENERATE(1000, 4096);
// 		auto seed = 42;
//
// 		std::string pattern = RandomBitString(size, seed);
// 		DynamicBitset bs(size);
//
// 		for (std::size_t i = 0; i < size; ++i)
// 		{
// 			bs.Set(i, pattern[i] == '1');
// 		}
//
// 		VerifyBitset(bs, pattern);
// 	}
// }
