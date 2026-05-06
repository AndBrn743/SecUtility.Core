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


