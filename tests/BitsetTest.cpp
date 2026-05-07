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

#include "cmake-build-debug-wsl-gcc/_deps/catch2-src/src/catch2/catch_template_test_macros.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <sstream>
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
		}
		{
			Bitset<64> bs;
			STATIC_REQUIRE(bs.Size() == 64);
		}
		{
			Bitset<32> bs;
			STATIC_REQUIRE(bs.Size() == 32);
		}
		{
			Bitset<1024> bs;
			STATIC_REQUIRE(bs.Size() == 1024);
		}
		{
			Bitset<264> bs;
			STATIC_REQUIRE(bs.Size() == 264);
		}
		{
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

		STATIC_REQUIRE(std::is_same_v<decltype(std::as_const(bs)[0]), bool>);
		STATIC_REQUIRE_FALSE(std::is_same_v<decltype(bs[0]), bool>);

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
		{
			Bitset<125> bs{true};

			bs.Reset(0).Reset(1).Reset(2);
			REQUIRE_FALSE(bs[0]);
			REQUIRE_FALSE(bs[1]);
			REQUIRE_FALSE(bs[2]);

			bs.Set(0).Set(1);
			REQUIRE_FALSE(!bs[0]);
			REQUIRE_FALSE(!bs[1]);
			REQUIRE_FALSE(bs[2]);

			bs.Flip(2).Flip(3);
			REQUIRE_FALSE(!bs[2]);
			REQUIRE_FALSE(bs[3]);
		}
	}
}


// =============================================================================
//  Test Iterators
// =============================================================================
TEST_CASE("Iterators", "[bitset][iterator]")
{
	SECTION("Forward iteration - empty bitset")
	{
		{
			Bitset<0> bs;
			REQUIRE(bs.begin() == bs.end());
			REQUIRE(bs.cbegin() == bs.cend());
		}
		{
			DynamicBitset bs;
			REQUIRE(bs.begin() == bs.end());
			REQUIRE(bs.cbegin() == bs.cend());
		}
	}

	SECTION("Forward iteration - Bitset<N>")
	{
		Bitset<16> bs;

		// Set some bits
		bs.Set(0).Set(3).Set(7).Set(15);

		std::vector<bool> collected;
		for (auto&& b : bs)
		{
			collected.push_back(b);
		}

		REQUIRE(collected.size() == 16);
		REQUIRE(collected[0] == true);
		REQUIRE(collected[1] == false);
		REQUIRE(collected[2] == false);
		REQUIRE(collected[3] == true);
		REQUIRE(collected[7] == true);
		REQUIRE(collected[15] == true);
	}

	SECTION("Forward iteration - range-based for loop")
	{
		const std::size_t size = 32;
		DynamicBitset bs(size);

		// Set alternating bits
		for (std::size_t i = 0; i < size; i += 2)
		{
			bs.Set(i);
		}

		std::size_t index = 0;
		for (bool bit : bs)
		{
			REQUIRE(bit == (index % 2 == 0));
			++index;
		}

		REQUIRE(index == size);
	}

	SECTION("Const iteration - cbegin/cend")
	{
		const Bitset<8> bs = []()
		{
			Bitset<8> tmp;
			tmp.Set(1).Set(3).Set(5).Set(7);
			return tmp;
		}();

		std::vector<bool> collected;
		for (bool b : std::as_const(bs))
		{
			collected.push_back(b);
		}

		REQUIRE(collected.size() == 8);
		REQUIRE(collected[0] == false);
		REQUIRE(collected[1] == true);
		REQUIRE(collected[2] == false);
		REQUIRE(collected[3] == true);
		REQUIRE(collected[4] == false);
		REQUIRE(collected[5] == true);
		REQUIRE(collected[6] == false);
		REQUIRE(collected[7] == true);
	}

	SECTION("Const iterator from const reference")
	{
		Bitset<10> bs;
		bs.Set(2).Set(5).Set(9);

		const Bitset<10>& cbs = bs;
		std::vector<bool> collected;

		for (bool cb : cbs)
		{
			collected.push_back(cb);
		}

		REQUIRE(collected.size() == 10);
		REQUIRE(collected[2] == true);
		REQUIRE(collected[5] == true);
		REQUIRE(collected[9] == true);
		REQUIRE(collected[0] == false);
		REQUIRE(collected[1] == false);
	}

	SECTION("Reverse iteration - rbegin/rend")
	{
		DynamicBitset bs(8);
		bs.Set(0).Set(1).Set(2);

		std::vector<bool> collected;
		for (auto it = bs.rbegin(); it != bs.rend(); ++it)
		{
			collected.push_back(*it);
		}

		// Reverse iteration should give us: index 7, 6, 5, 4, 3, 2, 1, 0
		REQUIRE(collected.size() == 8);
		REQUIRE(collected[0] == false);  // index 7
		REQUIRE(collected[1] == false);  // index 6
		REQUIRE(collected[2] == false);  // index 5
		REQUIRE(collected[3] == false);  // index 4
		REQUIRE(collected[4] == false);  // index 3
		REQUIRE(collected[5] == true);   // index 2
		REQUIRE(collected[6] == true);   // index 1
		REQUIRE(collected[7] == true);   // index 0
	}

	SECTION("Const reverse iteration - rcbegin/rcend")
	{
		const Bitset<5> bs = []()
		{
			Bitset<5> tmp;
			tmp.Set(0).Set(4);
			return tmp;
		}();

		std::vector<bool> collected;
		for (auto it = bs.crbegin(); it != bs.crend(); ++it)
		{
			collected.push_back(*it);
		}

		REQUIRE(collected.size() == 5);
		REQUIRE(collected[0] == true);   // index 4
		REQUIRE(collected[1] == false);  // index 3
		REQUIRE(collected[2] == false);  // index 2
		REQUIRE(collected[3] == false);  // index 1
		REQUIRE(collected[4] == true);   // index 0
	}

	SECTION("Reverse iteration from const reference")
	{
		DynamicBitset bs(6);
		bs.Set(1).Set(3).Set(5);

		const DynamicBitset& cbs = bs;
		std::vector<bool> collected;

		for (auto it = cbs.rbegin(); it != cbs.rend(); ++it)
		{
			collected.push_back(*it);
		}

		REQUIRE(collected.size() == 6);
		REQUIRE(collected[0] == true);   // index 5
		REQUIRE(collected[1] == false);  // index 4
		REQUIRE(collected[2] == true);   // index 3
		REQUIRE(collected[3] == false);  // index 2
		REQUIRE(collected[4] == true);   // index 1
		REQUIRE(collected[5] == false);  // index 0
	}

	SECTION("Iterator with std:: algorithms")
	{
		Bitset<10> bs;
		bs.Set(2).Set(5).Set(8);

		// std::count
		const auto count = std::count(bs.begin(), bs.end(), true);
		REQUIRE(count == 3);

		const auto zeroCount = std::count(bs.begin(), bs.end(), false);
		REQUIRE(zeroCount == 7);

		// std::find
		const auto it = std::find(bs.begin(), bs.end(), true);
		REQUIRE(it != bs.end());
		REQUIRE(*it == true);

		// std::find on const bitset
		const auto& cbs = bs;
		const auto cit = std::find(cbs.begin(), cbs.end(), false);
		REQUIRE(cit != cbs.end());
		REQUIRE(*cit == false);
	}

	SECTION("std::reverse_iterator with std::find")
	{
		DynamicBitset bs(8);
		bs.Set(3).Set(5).Set(7);

		// Find last true using reverse iterator
		auto rit = std::find(bs.rbegin(), bs.rend(), true);
		REQUIRE(rit != bs.rend());
		REQUIRE(*rit == true);  // Should find index 7 first in reverse

		++rit;
		REQUIRE(*rit == false);  // index 6
		++rit;
		REQUIRE(*rit == true);  // index 5
	}

	SECTION("Iterator compatibility with std::vector constructor")
	{
		Bitset<12> bs;
		bs.Set(0).Set(2).Set(4).Set(6).Set(8).Set(10);

		std::vector<bool> vec(bs.begin(), bs.end());

		REQUIRE(vec.size() == 12);
		for (std::size_t i = 0; i < 12; ++i)
		{
			REQUIRE(vec[i] == bs[i]);
		}
	}

	SECTION("Single element bitset iteration")
	{
		{
			Bitset<1> bsTrue{true};
			REQUIRE(*bsTrue.begin() == true);
			REQUIRE(*bsTrue.cbegin() == true);
			REQUIRE(*bsTrue.rbegin() == true);

			Bitset<1> bsFalse{false};
			REQUIRE(*bsFalse.begin() == false);
			REQUIRE(*bsFalse.cbegin() == false);
			REQUIRE(*bsFalse.rbegin() == false);
		}
		{
			DynamicBitset bs(1, true);
			REQUIRE(*bs.begin() == true);
			REQUIRE(*bs.cbegin() == true);
			REQUIRE(*bs.rbegin() == true);
		}
	}

	SECTION("Large bitset iteration")
	{
		const std::size_t size = 1000;
		DynamicBitset bs(size);

		// Set every 7th bit
		for (std::size_t i = 0; i < size; i += 7)
		{
			bs.Set(i);
		}

		std::size_t count = 0;
		for (const bool bit : bs)
		{
			if (bit)
			{
				++count;
			}
		}

		// Should have ceil(1000/7) = 143 bits set
		REQUIRE(count == 143);
	}

	SECTION("Iterator distance and arithmetic")
	{
		Bitset<20> bs;
		bs.Set(5).Set(10).Set(15);

		auto it1 = bs.begin();
		auto it2 = bs.begin() + 5;

		REQUIRE(*it2 == true);

		auto dist = it2 - it1;
		REQUIRE(dist == 5);
	}
}

TEST_CASE("BitsetNotExpr")
{
	const std::size_t size = GENERATE(0, 1, 4, 42, 69, 73, 423);
	const auto seed = GENERATE(0, 1, 42, 69, 73, 420, 4242, 6969, 66872);

	const auto bitString = RandomBitString(size, seed);
	DynamicBitset bs(size);
	std::transform(bitString.begin(), bitString.end(), bs.begin(), [](const char c) { return c != '0'; });

	auto nbs = ~bs;
	CHECK(nbs.Size() == bs.Size());

	for (std::size_t i = 0; i < size; ++i)
	{
		CHECK(nbs[i] != bs[i]);
		// nbs[i] = true;  // won't compile by design
		// nbs[i].Flip();  // won't compile by design
	}
}

// =============================================================================
//  Test BitsetSegmentExpr via Segment(), Leading(), Trailing()
// =============================================================================
TEST_CASE("BitsetSegmentExpr - basic functionality", "[bitset][segment]")
{
	SECTION("Segment() - basic read access")
	{
		Bitset<365> bs;

		const auto seed = GENERATE(42, 69, 73);
		std::mt19937_64 rng(seed);


		for (std::size_t i = 0; i < 100; ++i)
		{
			bs.Set(i, static_cast<bool>(rng() & 1u));
		}

		const std::size_t start = GENERATE(0, 42, 69, 73);
		const std::size_t size = GENERATE(0, 42, 69, 73, 240);

		auto seg = bs.Segment(start, size);
		REQUIRE(seg.Size() == size);

		for (std::size_t i = 0; i < size; ++i)
		{
			REQUIRE(seg[i] == bs[i + start]);
		}
		REQUIRE_THROWS(seg[size]);
	}

	SECTION("Segment() - write access modifies parent")
	{
		DynamicBitset bs(64);

		// Create segment covering middle 32 bits
		auto seg = bs.Segment(16, 32);

		// Set bits through segment
		for (std::size_t i = 0; i < 32; ++i)
		{
			seg.Set(i);
		}

		// Verify through parent
		for (std::size_t i = 0; i < 64; ++i)
		{
			REQUIRE(bs[i] == (i >= 16 && i < 16 + 32));
		}

		// Reset bits through segment
		for (std::size_t i = 0; i < 32; i += 2)
		{
			seg.Reset(i);
		}

		// Verify alternating pattern in parent
		for (std::size_t i = 16; i < 48; ++i)
		{
			REQUIRE(bs[i] == ((i - 16) % 2 == 1));
		}
	}

	SECTION("Segment() - iteration")
	{
		Bitset<80> bs;

		// Set every 3rd bit
		for (std::size_t i = 0; i < 80; i += 3)
		{
			bs.Set(i);
		}

		// Iterate over segment [10:50]
		auto seg = bs.Segment(10, 40);

		std::size_t count = 0;
		for (bool bit : seg)
		{
			if (bit)
			{
				++count;
			}
		}

		// Count bits set in range [10:50] where every 3rd bit is set
		std::size_t expected = 0;
		for (std::size_t i = 10; i < 50; ++i)
		{
			if (i % 3 == 0)
			{
				++expected;
			}
		}
		REQUIRE(count == expected);
	}

	SECTION("Segment() - const segment from const bitset")
	{
		const Bitset<50> bs = []
		{
			Bitset<50> tmp;
			for (std::size_t i = 0; i < 50; ++i)
			{
				tmp.Set(i, i % 3 == 0);
			}
			return tmp;
		}();

		// Segment should be const
		auto seg = bs.Segment(10, 30);

		// Should be able to read
		REQUIRE(seg.Size() == 30);
		for (std::size_t i = 0; i < 30; ++i)
		{
			REQUIRE(seg[i] == ((i + 10) % 3 == 0));
		}

		// This should NOT compile - const segment can't modify
		// seg.Set(0);  // Uncomment to test
	}

	SECTION("Leading() - get last N bits")
	{
		Bitset<100> bs;

		// Set last 20 bits
		for (std::size_t i = 80; i < 100; ++i)
		{
			bs.Set(i);
		}

		auto leading = bs.Leading(20);
		REQUIRE(leading.Size() == 20);

		// All should be true
		for (std::size_t i = 0; i < 20; ++i)
		{
			REQUIRE(leading[i] == true);
		}

		// STL algorithm should work
		REQUIRE(std::all_of(leading.begin(), leading.end(), [](const bool b) { return b; }));

		// Modify through leading
		leading.Reset(0);

		// Should affect bit 80 in parent
		REQUIRE_FALSE(bs[80]);
		REQUIRE(bs[81]);
	}

	SECTION("Trailing() - get first N bits")
	{
		DynamicBitset bs(100);

		// Set first 30 bits
		std::for_each_n(bs.begin(), 30, [](auto&& b) { b = true; });

		auto trailing = bs.Trailing(30);
		REQUIRE(trailing.Size() == 30);

		// All should be true
		for (std::size_t i = 0; i < 30; ++i)
		{
			REQUIRE(trailing[i] == true);
		}

		// Modify through trailing
		trailing.Reset(15);

		// Should affect bit 15 in parent
		REQUIRE(bs[14]);
		REQUIRE_FALSE(bs[15]);
		REQUIRE(bs[16]);
	}

	SECTION("Leading() and Trailing() with empty/small ranges")
	{
		Bitset<64> bs;
		bs.Set(0).Set(63);

		// Leading(0) should be empty
		auto leadingEmpty = bs.Leading(0);
		REQUIRE(leadingEmpty.Size() == 0);
		REQUIRE(leadingEmpty.begin() == leadingEmpty.end());

		// Trailing(0) should be empty
		auto trailingEmpty = bs.Trailing(0);
		REQUIRE(trailingEmpty.Size() == 0);
		REQUIRE(trailingEmpty.begin() == trailingEmpty.end());

		// Leading(1) should have only last bit
		auto leadingOne = bs.Leading(1);
		REQUIRE(leadingOne.Size() == 1);
		REQUIRE(leadingOne[0] == true);

		// Trailing(1) should have only first bit
		auto trailingOne = bs.Trailing(1);
		REQUIRE(trailingOne.Size() == 1);
		REQUIRE(trailingOne[0] == true);
	}
}

TEST_CASE(".ToString() and operator<<")
{
	const std::size_t size = GENERATE(0, 1, 42, 69, 73, 420);
	const auto seed = GENERATE(0, 1, 42, 69, 73, 420);

	const auto bitString = RandomBitString(size, seed);
	DynamicBitset bs(size);
	std::transform(bitString.rbegin(), bitString.rend(), bs.begin(), [](const char c) { return c != '0'; });

	const auto msbfString = bs.ToString();
	CHECK(msbfString == bitString);
	{
		std::ostringstream msbfOss;
		msbfOss << bs;
		CHECK(msbfOss.str() == msbfString);
	}
	{
		std::ostringstream msbfOss;
		msbfOss << MsbFirst << bs;
		CHECK(msbfOss.str() == msbfString);
	}
	{
		std::ostringstream msbfOss;
		msbfOss << LsbFirst << MsbFirst << bs;
		CHECK(msbfOss.str() == msbfString);
	}

	auto string = bs.ToString(false);
	{
		std::ostringstream msbfOss;
		msbfOss << LsbFirst << bs;
		CHECK(msbfOss.str() == string);
	}
	{
		std::ostringstream msbfOss;
		msbfOss << LsbFirst << MsbFirst << LsbFirst << bs;
		CHECK(msbfOss.str() == string);
	}
	std::reverse(string.begin(), string.end());
	CHECK(string == bitString);
}

TEST_CASE("SetAll, ResetAll, and FlipAll")
{
	const std::size_t size = GENERATE(42, 69, 73, 420);
	const auto seed = GENERATE(0, 1, 42, 69, 73, 420);

	const auto bitString = RandomBitString(size, seed);
	DynamicBitset bs(size);
	std::transform(bitString.rbegin(), bitString.rend(), bs.begin(), [](const char c) { return c != '0'; });

	SECTION("SetAll(true)")
	{
		bs.SetAll(true);
		CHECK(std::all_of(bs.begin(), bs.end(), [](const bool b) { return b; }));
	}

	SECTION("SetAll(false)")
	{
		bs.SetAll(false);
		CHECK(std::none_of(bs.begin(), bs.end(), [](const bool b) { return b; }));
	}

	SECTION("FlipAll()")
	{
		bs.FlipAll();

		auto bi = bs.begin();
		auto sri = bitString.rbegin();
		for (/* NO CODE */; sri != bitString.rend(); ++sri, ++bi)
		{
			CHECK(*bi != *sri);
		}
	}

	auto seg0 = bs.Segment(0, 10);
	auto seg1 = bs.Segment(10, size - 10 - 15);
	auto seg2 = bs.Segment(size - 15, 15);

	SECTION("seg.SetAll(true)")
	{
		const auto ss0 = seg0.ToString();
		const auto ss2 = seg2.ToString();
		seg1.SetAll(true);
		REQUIRE(seg0.ToString() == ss0);
		REQUIRE(seg2.ToString() == ss2);
		CHECK(std::all_of(seg1.begin(), seg1.end(), [](const bool b) { return b; }));
	}

	SECTION("seg.ResetAll()")
	{
		const auto ss0 = seg0.ToString();
		const auto ss2 = seg2.ToString();
		seg1.ResetAll();
		REQUIRE(seg0.ToString() == ss0);
		REQUIRE(seg2.ToString() == ss2);
		CHECK(std::none_of(seg1.begin(), seg1.end(), [](const bool b) { return b; }));
	}

	SECTION("seg.FlipAll()")
	{
		const auto ss0 = seg0.ToString();
		const auto ss2 = seg2.ToString();
		seg1.FlipAll();
		REQUIRE(seg0.ToString() == ss0);
		REQUIRE(seg2.ToString() == ss2);

		auto bi = seg1.begin();
		auto sri = bitString.rbegin() + 15;
		for (/* NO CODE */; bi != seg1.end(); ++sri, ++bi)
		{
			CHECK(*bi != *sri);
		}
	}
}

TEST_CASE("IsAllOnes, IsAllZeros, HasOnes, and HasZeros")
{
	SECTION("Known samples")
	{
		{
			Bitset<0> bs{};
			CHECK_FALSE(bs.IsAllOnes());
			CHECK_FALSE(bs.IsAllZeros());
			CHECK_FALSE(bs.HasOnes());
			CHECK_FALSE(bs.HasZeros());
		}
		{
			DynamicBitset bs{};
			CHECK_FALSE(bs.IsAllOnes());
			CHECK_FALSE(bs.IsAllZeros());
			CHECK_FALSE(bs.HasOnes());
			CHECK_FALSE(bs.HasZeros());
		}
		{
			Bitset<3> bs{};
			CHECK_FALSE(bs.IsAllOnes());
			CHECK(bs.IsAllZeros());
			CHECK_FALSE(bs.HasOnes());
			CHECK(bs.HasZeros());
		}
		{
			Bitset<3> bs{true};
			CHECK(bs.IsAllOnes());
			CHECK_FALSE(bs.IsAllZeros());
			CHECK(bs.HasOnes());
			CHECK_FALSE(bs.HasZeros());
		}
		{
			Bitset<3> bs{true};
			bs[0].Flip();
			CHECK_FALSE(bs.IsAllOnes());
			CHECK_FALSE(bs.IsAllZeros());
			CHECK(bs.HasOnes());
			CHECK(bs.HasZeros());
		}
		{
			DynamicBitset bs{200};
			CHECK_FALSE(bs.IsAllOnes());
			CHECK(bs.IsAllZeros());
			CHECK_FALSE(bs.HasOnes());
			CHECK(bs.HasZeros());
		}
		{
			DynamicBitset bs{200, true};
			CHECK(bs.IsAllOnes());
			CHECK_FALSE(bs.IsAllZeros());
			CHECK(bs.HasOnes());
			CHECK_FALSE(bs.HasZeros());
		}
		{
			DynamicBitset bs{200, true};
			bs.Flip(42).Flip(69);
			CHECK_FALSE(bs.IsAllOnes());
			CHECK_FALSE(bs.IsAllZeros());
			CHECK(bs.HasOnes());
			CHECK(bs.HasZeros());

			{
				auto seg = bs.Trailing(4);
				CHECK(seg.IsAllOnes());
				CHECK_FALSE(seg.IsAllZeros());
				CHECK(seg.HasOnes());
				CHECK_FALSE(seg.HasZeros());
			}
			{
				auto seg = bs.Trailing(42);
				CHECK(seg.IsAllOnes());
				CHECK_FALSE(seg.IsAllZeros());
				CHECK(seg.HasOnes());
				CHECK_FALSE(seg.HasZeros());
			}
			{
				auto seg = bs.Trailing(43);
				CHECK_FALSE(seg.IsAllOnes());
				CHECK_FALSE(seg.IsAllZeros());
				CHECK(seg.HasOnes());
				CHECK(seg.HasZeros());
			}
			{
				auto seg = bs.Segment(42, 1);
				CHECK_FALSE(seg.IsAllOnes());
				CHECK(seg.IsAllZeros());
				CHECK_FALSE(seg.HasOnes());
				CHECK(seg.HasZeros());
			}

			{
				auto seg = std::as_const(bs).Trailing(42);
				CHECK(seg.IsAllOnes());
				CHECK_FALSE(seg.IsAllZeros());
				CHECK(seg.HasOnes());
				CHECK_FALSE(seg.HasZeros());
			}
			{
				auto seg = std::as_const(bs).Trailing(43);
				CHECK_FALSE(seg.IsAllOnes());
				CHECK_FALSE(seg.IsAllZeros());
				CHECK(seg.HasOnes());
				CHECK(seg.HasZeros());
			}
			{
				auto seg = std::as_const(bs).Segment(42, 1);
				CHECK_FALSE(seg.IsAllOnes());
				CHECK(seg.IsAllZeros());
				CHECK_FALSE(seg.HasOnes());
				CHECK(seg.HasZeros());
			}
		}
	}

	SECTION("Random samples")
	{
		const std::size_t size = GENERATE(42, 69, 73, 420);
		const auto seed = GENERATE(0, 1, 42, 69, 73, 420, 4242, 6969, 66872);

		const auto bitString = RandomBitString(size, seed);
		DynamicBitset bs(size);
		std::transform(bitString.rbegin(), bitString.rend(), bs.begin(), [](const char c) { return c != '0'; });

		{
			auto seg = bs.Segment(bs.Size() / 2, 4);
			CHECK(seg.IsAllOnes() == std::all_of(seg.begin(), seg.end(), [](const bool b) { return b; }));
			CHECK(seg.IsAllZeros() == std::none_of(seg.begin(), seg.end(), [](const bool b) { return b; }));
		}
		{
			auto seg = std::as_const(bs).Segment(bs.Size() / 2, 4);
			CHECK(seg.IsAllOnes() == std::all_of(seg.begin(), seg.end(), [](const bool b) { return b; }));
			CHECK(seg.IsAllZeros() == std::none_of(seg.begin(), seg.end(), [](const bool b) { return b; }));
		}
	}
}

TEST_CASE("OneCount and ZeroCount")
{
	const std::size_t size = GENERATE(0, 1, 4, 42, 69, 73, 420);
	const auto seed = GENERATE(0, 1, 42, 69, 73, 420, 4242, 6969, 66872);

	const auto bitString = RandomBitString(size, seed);
	DynamicBitset bs(size);
	std::transform(bitString.rbegin(), bitString.rend(), bs.begin(), [](const char c) { return c != '0'; });

	const auto oneCount = bs.OneCount();
	const auto zeroCount = bs.ZeroCount();

	CHECK(oneCount
	      == static_cast<std::size_t>(
	              std::count_if(bitString.begin(), bitString.end(), [](const char c) { return c != '0'; })));
	CHECK(zeroCount
	      == static_cast<std::size_t>(
	              std::count_if(bitString.begin(), bitString.end(), [](const char c) { return c == '0'; })));
}

TEST_CASE("operator==")
{
	{
		constexpr char bitString[] = "0110001001100001111011000100110111111010110101110101101100011101"
		                             "0110001000111011100010011011100110111110110000010011000101110010"
		                             "0010110000100000000101100000101110001111010001001101111101011101"
		                             "0000100000011101110011111001100010111000111011100110101110101110"
		                             "1010110001000010100001110100100111011000100111011100110101011101";
		DynamicBitset bs(std::string_view(bitString).size());
		std::transform(
		        std::begin(bitString), std::end(bitString) - 1, std::begin(bs), [](const char c) { return c != '0'; });

		CHECK(bs.Segment(20, 8) == bs.Segment(1, 8));
		CHECK(bs.Segment(0, 9) == bs.Segment(64, 9));
		CHECK(bs.Segment(0, 10) != bs.Segment(64, 10));
		CHECK(bs.Segment(0, 10) != bs.Segment(64, 9));

		CHECK(bs.Segment(0, 64) != bs.Segment(15, 64));
		CHECK(bs.Segment(0, 165) != bs.Segment(15, 165));
		CHECK(bs.Segment(15, 64) != bs.Segment(0, 64));
		CHECK(bs.Segment(15, 165) != bs.Segment(0, 165));
	}
	{
		DynamicBitset bs{365};

		for (std::size_t i = 0; i < bs.Size(); ++i)
		{
			if (i % 5 == 0)
			{
				bs.Flip(i);
			}
		}

		CHECK(bs.Segment(0, 128) == bs.Segment(100, 128));
		CHECK(bs.Segment(100, 128) == bs.Segment(0, 128));

		CHECK(bs.Segment(0, 135) == bs.Segment(100, 135));
		CHECK(bs.Segment(100, 135) == bs.Segment(0, 135));

		CHECK(bs.Segment(1, 135) != bs.Segment(100, 135));
		CHECK(bs.Segment(100, 135) != bs.Segment(1, 135));
	}
}

TEST_CASE("TrailingZeroCount and TrailingOneCount")
{
	const std::size_t size = GENERATE(0, 1, 4, 42, 69, 73, 423);
	const auto seed = GENERATE(0, 1, 42, 69, 73, 420, 4242, 6969, 66872);

	const auto bitString = RandomBitString(size, seed);
	DynamicBitset bs(size);
	std::transform(bitString.begin(), bitString.end(), bs.begin(), [](const char c) { return c != '0'; });

	SECTION("Random")
	{
		CHECK(bs.TrailingZeroCount()
		      == static_cast<std::size_t>(std::distance(
		              bitString.cbegin(),
		              std::find_if(bitString.cbegin(), bitString.cend(), [](const char c) { return c != '0'; }))));

		CHECK(bs.TrailingOneCount()
		      == static_cast<std::size_t>(std::distance(
		              bitString.cbegin(),
		              std::find_if(bitString.cbegin(), bitString.cend(), [](const char c) { return c == '0'; }))));

		CHECK(bs.IndexOfFirstOne()
		      == static_cast<std::size_t>(std::distance(
		              bitString.cbegin(),
		              std::find_if(bitString.cbegin(), bitString.cend(), [](const char c) { return c != '0'; }))));

		CHECK(bs.IndexOfFirstZero()
		      == static_cast<std::size_t>(std::distance(
		              bitString.cbegin(),
		              std::find_if(bitString.cbegin(), bitString.cend(), [](const char c) { return c == '0'; }))));
	}

	SECTION("Random with modification")
	{
		const std::size_t t = GENERATE(32, 63, 64, 65, 165, 235, 400);

		if (bs.Size() > t)
		{
			bs.Trailing(t).ResetAll();
			const auto tzc = bs.TrailingZeroCount();
			CHECK(tzc
			      == static_cast<std::size_t>(std::distance(
			              bs.cbegin(), std::find_if(bs.cbegin(), bs.cend(), [](const bool b) { return b; }))));

			bs.Trailing(t).SetAll();
			const auto toc = bs.TrailingOneCount();
			CHECK(toc
			      == static_cast<std::size_t>(std::distance(
			              bs.cbegin(), std::find_if(bs.cbegin(), bs.cend(), [](const bool b) { return !b; }))));
		}

		if (bs.Size() > 8 && bs.Size() > t + 8)
		{
			auto seg = bs.Segment(8, t);

			bs.Trailing(t).ResetAll();
			const auto tzc = seg.TrailingZeroCount();

			CHECK(tzc
			      == static_cast<std::size_t>(std::distance(
			              seg.cbegin(), std::find_if(seg.cbegin(), seg.cend(), [](const bool b) { return b; }))));

			bs.Trailing(t).SetAll();
			const auto toc = seg.TrailingOneCount();
			CHECK(toc
			      == static_cast<std::size_t>(std::distance(
			              seg.cbegin(), std::find_if(seg.cbegin(), seg.cend(), [](const bool b) { return !b; }))));
		}
	}

	SECTION("Edge case")
	{
		bs.ResetAll();
		CHECK(bs.TrailingZeroCount() == bs.Size());
		CHECK(bs.TrailingOneCount() == 0);

		bs.SetAll();
		CHECK(bs.TrailingZeroCount() == 0);
		CHECK(bs.TrailingOneCount() == bs.Size());
	}
}

TEST_CASE("LeadingZeroCount and LeadingOneCount")
{
	const std::size_t size = GENERATE(0, 1, 4, 42, 69, 73, 423);
	const auto seed = GENERATE(0, 1, 42, 69, 73, 420, 4242, 6969, 66872);

	const auto bitString = RandomBitString(size, seed);
	DynamicBitset bs(size);
	std::transform(bitString.begin(), bitString.end(), bs.begin(), [](const char c) { return c != '0'; });

	SECTION("Random")
	{
		const auto tzc = bs.LeadingZeroCount();
		CHECK(tzc
		      == static_cast<std::size_t>(std::distance(
		              bitString.crbegin(),
		              std::find_if(bitString.crbegin(), bitString.crend(), [](const char c) { return c != '0'; }))));

		const auto toc = bs.LeadingOneCount();
		CHECK(toc
		      == static_cast<std::size_t>(std::distance(
		              bitString.crbegin(),
		              std::find_if(bitString.crbegin(), bitString.crend(), [](const char c) { return c == '0'; }))));
	}

	SECTION("Random with modification")
	{
		const std::size_t t = GENERATE(32, 63, 64, 65, 165, 235, 400);

		if (bs.Size() > t)
		{
			bs.Leading(t).ResetAll();
			const auto tzc = bs.LeadingZeroCount();
			CHECK(tzc
			      == static_cast<std::size_t>(std::distance(
			              bs.crbegin(), std::find_if(bs.crbegin(), bs.crend(), [](const bool b) { return b; }))));

			bs.Leading(t).SetAll();
			const auto toc = bs.LeadingOneCount();
			CHECK(toc
			      == static_cast<std::size_t>(std::distance(
			              bs.crbegin(), std::find_if(bs.crbegin(), bs.crend(), [](const bool b) { return !b; }))));
		}

		if (bs.Size() > 8 && bs.Size() > t + 8)
		{
			auto seg = bs.Segment(bs.Size() - 8 - t, t);

			bs.Leading(t).ResetAll();
			const auto tzc = seg.LeadingZeroCount();

			CHECK(tzc
			      == static_cast<std::size_t>(std::distance(
			              seg.crbegin(), std::find_if(seg.crbegin(), seg.crend(), [](const bool b) { return b; }))));

			bs.Leading(t).SetAll();
			const auto toc = seg.LeadingOneCount();
			CHECK(toc
			      == static_cast<std::size_t>(std::distance(
			              seg.crbegin(), std::find_if(seg.crbegin(), seg.crend(), [](const bool b) { return !b; }))));
		}
	}

	SECTION("Edge case")
	{
		bs.ResetAll();
		CHECK(bs.LeadingZeroCount() == bs.Size());
		CHECK(bs.LeadingOneCount() == 0);
		CHECK(bs.IndexOfFirstZero() == 0);
		CHECK(bs.IndexOfFirstOne() == bs.Size());
		CHECK(bs.IndexOfLastZero() == (bs.Size() == 0 ? 0 : bs.Size() - 1));
		CHECK(bs.IndexOfLastOne() == bs.Size());

		bs.SetAll();
		CHECK(bs.LeadingZeroCount() == 0);
		CHECK(bs.LeadingOneCount() == bs.Size());
		CHECK(bs.IndexOfFirstZero() == bs.Size());
		CHECK(bs.IndexOfFirstOne() == 0);
		CHECK(bs.IndexOfLastZero() == bs.Size());
		CHECK(bs.IndexOfLastOne() == (bs.Size() == 0 ? 0 : bs.Size() - 1));
	}
}

TEST_CASE("operator&=, operator|=, and operator^=")
{
	const std::size_t size = GENERATE(42, 69, 73, 423);
	const auto seed = GENERATE(0, 1, 42, 69, 73, 420, 4242, 6969, 66872);

	const auto bitString0 = RandomBitString(size, seed);
	DynamicBitset bs0(size);
	std::transform(bitString0.begin(), bitString0.end(), bs0.begin(), [](const char c) { return c != '0'; });

	const auto bitString1 = RandomBitString(size, seed + 1);
	DynamicBitset bs1(size);
	std::transform(bitString1.begin(), bitString1.end(), bs1.begin(), [](const char c) { return c != '0'; });

	auto seg0 = bs0.Trailing(12);
	auto seg2 = bs0.Leading(19);

	SECTION("operator&=")
	{
		{
			DynamicBitset bs2 = bs0;
			bs2.Segment(12, size - seg0.Size() - seg2.Size()) &= bs1.Segment(12, size - seg0.Size() - seg2.Size());

			REQUIRE(seg0.ToString() == bs2.Trailing(12).ToString());
			REQUIRE(seg2.ToString() == bs2.Leading(19).ToString());

			for (std::size_t i = 0; i < size - seg0.Size() - seg2.Size(); ++i)
			{
				CHECK(bs2[12 + i] == (bs0[12 + i] && bs1[12 + i]));
			}
		}
		{
			DynamicBitset bs2 = bs0;
			bs2.Segment(12, size - seg0.Size() - seg2.Size()) &= bs1.Segment(12 + 9, size - seg0.Size() - seg2.Size());

			REQUIRE(seg0.ToString() == bs2.Trailing(12).ToString());
			REQUIRE(seg2.ToString() == bs2.Leading(19).ToString());

			for (std::size_t i = 0; i < size - seg0.Size() - seg2.Size(); ++i)
			{
				CHECK(bs2[12 + i] == (bs0[12 + i] && bs1[12 + 9 + i]));
			}
		}
		{
			DynamicBitset bs2 = bs0;
			bs2.Segment(12, size - seg0.Size() - seg2.Size()) &= bs1.Segment(12 - 9, size - seg0.Size() - seg2.Size());

			REQUIRE(seg0.ToString() == bs2.Trailing(12).ToString());
			REQUIRE(seg2.ToString() == bs2.Leading(19).ToString());

			for (std::size_t i = 0; i < size - seg0.Size() - seg2.Size(); ++i)
			{
				CHECK(bs2[12 + i] == (bs0[12 + i] && bs1[12 - 9 + i]));
			}
		}
	}

	SECTION("operator|=")
	{
		{
			DynamicBitset bs2 = bs0;
			bs2.Segment(12, size - seg0.Size() - seg2.Size()) |= bs1.Segment(12, size - seg0.Size() - seg2.Size());

			REQUIRE(seg0.ToString() == bs2.Trailing(12).ToString());
			REQUIRE(seg2.ToString() == bs2.Leading(19).ToString());

			for (std::size_t i = 0; i < size - seg0.Size() - seg2.Size(); ++i)
			{
				CHECK(bs2[12 + i] == (bs0[12 + i] || bs1[12 + i]));
			}
		}
		{
			DynamicBitset bs2 = bs0;
			bs2.Segment(12, size - seg0.Size() - seg2.Size()) |= bs1.Segment(12 + 9, size - seg0.Size() - seg2.Size());

			REQUIRE(seg0.ToString() == bs2.Trailing(12).ToString());
			REQUIRE(seg2.ToString() == bs2.Leading(19).ToString());

			for (std::size_t i = 0; i < size - seg0.Size() - seg2.Size(); ++i)
			{
				CHECK(bs2[12 + i] == (bs0[12 + i] || bs1[12 + 9 + i]));
			}
		}
		{
			DynamicBitset bs2 = bs0;
			bs2.Segment(12, size - seg0.Size() - seg2.Size()) |= bs1.Segment(12 - 9, size - seg0.Size() - seg2.Size());

			REQUIRE(seg0.ToString() == bs2.Trailing(12).ToString());
			REQUIRE(seg2.ToString() == bs2.Leading(19).ToString());

			for (std::size_t i = 0; i < size - seg0.Size() - seg2.Size(); ++i)
			{
				CHECK(bs2[12 + i] == (bs0[12 + i] || bs1[12 - 9 + i]));
			}
		}
	}

	SECTION("operator^=")
	{
		{
			DynamicBitset bs2 = bs0;
			bs2.Segment(12, size - seg0.Size() - seg2.Size()) ^= bs1.Segment(12, size - seg0.Size() - seg2.Size());

			REQUIRE(seg0.ToString() == bs2.Trailing(12).ToString());
			REQUIRE(seg2.ToString() == bs2.Leading(19).ToString());

			for (std::size_t i = 0; i < size - seg0.Size() - seg2.Size(); ++i)
			{
				CHECK(bs2[12 + i] == (bs0[12 + i] != bs1[12 + i]));
			}
		}
		{
			DynamicBitset bs2 = bs0;
			bs2.Segment(12, size - seg0.Size() - seg2.Size()) ^= bs1.Segment(12 + 9, size - seg0.Size() - seg2.Size());

			REQUIRE(seg0.ToString() == bs2.Trailing(12).ToString());
			REQUIRE(seg2.ToString() == bs2.Leading(19).ToString());

			for (std::size_t i = 0; i < size - seg0.Size() - seg2.Size(); ++i)
			{
				CHECK(bs2[12 + i] == (bs0[12 + i] != bs1[12 + 9 + i]));
			}
		}
		{
			DynamicBitset bs2 = bs0;
			bs2.Segment(12, size - seg0.Size() - seg2.Size()) ^= bs1.Segment(12 - 9, size - seg0.Size() - seg2.Size());

			REQUIRE(seg0.ToString() == bs2.Trailing(12).ToString());
			REQUIRE(seg2.ToString() == bs2.Leading(19).ToString());

			for (std::size_t i = 0; i < size - seg0.Size() - seg2.Size(); ++i)
			{
				CHECK(bs2[12 + i] == (bs0[12 + i] != bs1[12 - 9 + i]));
			}
		}
	}
}

TEST_CASE("Ctors")
{
	static constexpr auto SetRandom = [](auto& bitset, const auto seed)
	{
		const auto bitString0 = RandomBitString(bitset.Size(), seed);
		std::transform(bitString0.begin(), bitString0.end(), bitset.begin(), [](const char c) { return c != '0'; });
	};

	const auto seed = GENERATE(0, 1, 42, 69, 73, 420, 4242, 6969, 66872);

	{
		Bitset<42> bs0{};
		SetRandom(bs0, seed);

		{
			Bitset<42> copy = Bitset<42>{bs0};
			CHECK(copy.ToString() == bs0.ToString());
		}

		{
			Bitset<42> copy = static_cast<Bitset<42>>(static_cast<BitsetBase<Bitset<42>>&>(bs0));
			CHECK(copy.ToString() == bs0.ToString());
		}

		{
			Bitset<42> copy = Bitset<42>{static_cast<BitsetBase<Bitset<42>>&>(bs0)};
			CHECK(copy.ToString() == bs0.ToString());
		}

		{
			Bitset<42> copy = static_cast<Bitset<42>>(static_cast<const BitsetBase<Bitset<42>>&>(bs0));
			CHECK(copy.ToString() == bs0.ToString());
		}

		{
			Bitset<42> copy = Bitset<42>{static_cast<const BitsetBase<Bitset<42>>&>(bs0)};
			CHECK(copy.ToString() == bs0.ToString());
		}

		{
			const auto expected = bs0.ToString();
			Bitset<42> copy = Bitset<42>{static_cast<BitsetBase<Bitset<42>>&&>(bs0)};
			CHECK(copy.ToString() == expected);
		}
	}

	{
		DynamicBitset bs0{42};
		SetRandom(bs0, seed);

		{
			DynamicBitset copy = DynamicBitset{bs0};
			CHECK(copy.ToString() == bs0.ToString());
		}

		{
			DynamicBitset copy = static_cast<DynamicBitset>(static_cast<BitsetBase<DynamicBitset>&>(bs0));
			CHECK(copy.ToString() == bs0.ToString());
		}

		{
			DynamicBitset copy = DynamicBitset{static_cast<BitsetBase<DynamicBitset>&>(bs0)};
			CHECK(copy.ToString() == bs0.ToString());
		}

		{
			DynamicBitset copy = static_cast<DynamicBitset>(static_cast<const BitsetBase<DynamicBitset>&>(bs0));
			CHECK(copy.ToString() == bs0.ToString());
		}

		{
			DynamicBitset copy = DynamicBitset{static_cast<const BitsetBase<DynamicBitset>&>(bs0)};
			CHECK(copy.ToString() == bs0.ToString());
		}

		{
			const auto expected = bs0.ToString();
			DynamicBitset copy = DynamicBitset{static_cast<BitsetBase<DynamicBitset>&&>(bs0)};
			CHECK(copy.ToString() == expected);
		}
	}



}

TEST_CASE("operator&, operator|, and operator^")
{
	static constexpr auto SetRandom = [](auto& bitset, const auto seed)
	{
		const auto bitString0 = RandomBitString(bitset.Size(), seed);
		std::transform(bitString0.begin(), bitString0.end(), bitset.begin(), [](const char c) { return c != '0'; });
	};

	const auto seed = GENERATE(0, 1, 42, 69, 73, 420, 4242, 6969, 66872);

	SECTION("operator&")
	{
		{
			Bitset<42> bs0{};
			SetRandom(bs0, seed);

			Bitset<42> bs1{};
			SetRandom(bs1, seed + 10);

			const auto bn = bs0 & bs1;
			STATIC_CHECK(std::is_same_v<std::decay_t<decltype(bn)>, Bitset<42>>);

			for (std::size_t i = 0; i < 42; ++i)
			{
				CHECK(bn[i] == (bs0[i] && bs1[i]));
			}
		}

		{
			Bitset<42> bs0{};
			SetRandom(bs0, seed);

			DynamicBitset bs1{42};
			SetRandom(bs1, seed + 10);

			const auto bn = bs0 & bs1;
			STATIC_CHECK(std::is_same_v<std::decay_t<decltype(bn)>, Bitset<42>>);

			for (std::size_t i = 0; i < 42; ++i)
			{
				CHECK(bn[i] == (bs0[i] && bs1[i]));
			}
		}

		{
			DynamicBitset bs0{42};
			SetRandom(bs0, seed);

			Bitset<42> bs1{};
			SetRandom(bs1, seed + 10);

			const auto bn = bs0 & bs1;
			STATIC_CHECK(std::is_same_v<std::decay_t<decltype(bn)>, Bitset<42>>);

			for (std::size_t i = 0; i < 42; ++i)
			{
				CHECK(bn[i] == (bs0[i] && bs1[i]));
			}
		}

		{
			DynamicBitset bs0{42};
			SetRandom(bs0, seed);

			DynamicBitset bs1{42};
			SetRandom(bs1, seed + 10);

			const auto bn = bs0 & bs1;
			STATIC_CHECK(std::is_same_v<std::decay_t<decltype(bn)>, DynamicBitset>);

			for (std::size_t i = 0; i < 42; ++i)
			{
				CHECK(bn[i] == (bs0[i] && bs1[i]));
			}
		}
	}

	SECTION("operator|")
	{
		{
			Bitset<42> bs0{};
			SetRandom(bs0, seed);

			Bitset<42> bs1{};
			SetRandom(bs1, seed + 10);

			const auto bn = bs0 | bs1;
			STATIC_CHECK(std::is_same_v<std::decay_t<decltype(bn)>, Bitset<42>>);

			for (std::size_t i = 0; i < 42; ++i)
			{
				CHECK(bn[i] == (bs0[i] || bs1[i]));
			}
		}

		{
			Bitset<42> bs0{};
			SetRandom(bs0, seed);

			DynamicBitset bs1{42};
			SetRandom(bs1, seed + 10);

			const auto bn = bs0 | bs1;
			STATIC_CHECK(std::is_same_v<std::decay_t<decltype(bn)>, Bitset<42>>);

			for (std::size_t i = 0; i < 42; ++i)
			{
				CHECK(bn[i] == (bs0[i] || bs1[i]));
			}
		}

		{
			DynamicBitset bs0{42};
			SetRandom(bs0, seed);

			Bitset<42> bs1{};
			SetRandom(bs1, seed + 10);

			const auto bn = bs0 | bs1;
			STATIC_CHECK(std::is_same_v<std::decay_t<decltype(bn)>, Bitset<42>>);

			for (std::size_t i = 0; i < 42; ++i)
			{
				CHECK(bn[i] == (bs0[i] || bs1[i]));
			}
		}

		{
			DynamicBitset bs0{42};
			SetRandom(bs0, seed);

			DynamicBitset bs1{42};
			SetRandom(bs1, seed + 10);

			const auto bn = bs0 | bs1;
			STATIC_CHECK(std::is_same_v<std::decay_t<decltype(bn)>, DynamicBitset>);

			for (std::size_t i = 0; i < 42; ++i)
			{
				CHECK(bn[i] == (bs0[i] || bs1[i]));
			}
		}
	}

	SECTION("operator^")
	{
		{
			Bitset<42> bs0{};
			SetRandom(bs0, seed);

			Bitset<42> bs1{};
			SetRandom(bs1, seed + 10);

			const auto bn = bs0 ^ bs1;
			STATIC_CHECK(std::is_same_v<std::decay_t<decltype(bn)>, Bitset<42>>);

			for (std::size_t i = 0; i < 42; ++i)
			{
				CHECK(bn[i] == (bs0[i] != bs1[i]));
			}
		}

		{
			Bitset<42> bs0{};
			SetRandom(bs0, seed);

			DynamicBitset bs1{42};
			SetRandom(bs1, seed + 10);

			const auto bn = bs0 ^ bs1;
			STATIC_CHECK(std::is_same_v<std::decay_t<decltype(bn)>, Bitset<42>>);

			for (std::size_t i = 0; i < 42; ++i)
			{
				CHECK(bn[i] == (bs0[i] != bs1[i]));
			}
		}

		{
			DynamicBitset bs0{42};
			SetRandom(bs0, seed);

			Bitset<42> bs1{};
			SetRandom(bs1, seed + 10);

			const auto bn = bs0 ^ bs1;
			STATIC_CHECK(std::is_same_v<std::decay_t<decltype(bn)>, Bitset<42>>);

			for (std::size_t i = 0; i < 42; ++i)
			{
				CHECK(bn[i] == (bs0[i] != bs1[i]));
			}
		}

		{
			DynamicBitset bs0{42};
			SetRandom(bs0, seed);

			DynamicBitset bs1{42};
			SetRandom(bs1, seed + 10);

			const auto bn = bs0 ^ bs1;
			STATIC_CHECK(std::is_same_v<std::decay_t<decltype(bn)>, DynamicBitset>);

			for (std::size_t i = 0; i < 42; ++i)
			{
				CHECK(bn[i] == (bs0[i] != bs1[i]));
			}
		}
	}
}

TEST_CASE("operator<<= and operator>>=")
{
	const std::size_t size = GENERATE(42, 69, 73, 423);
	const auto seed = GENERATE(0, 1, 42, 69, 73, 420, 4242, 6969, 66872);

	const auto bitString0 = RandomBitString(size, seed);
	DynamicBitset bs0(size);
	std::transform(bitString0.begin(), bitString0.end(), bs0.begin(), [](const char c) { return c != '0'; });

	auto seg0 = bs0.Trailing(12);
	auto seg2 = bs0.Leading(19);
	auto seg1 = bs0.Segment(12, bs0.Size() - seg0.Size() - seg2.Size());

	std::size_t shiftSize = GENERATE(0, 1, 9, 32, 63, 64, 65, 70, 128, 300, 450);

	SECTION("operator<<=")
	{
		DynamicBitset bs2 = bs0;
		auto seg = bs2.Segment(12, size - seg0.Size() - seg2.Size());
		seg <<= shiftSize;

		REQUIRE(seg0.ToString() == bs2.Trailing(12).ToString());
		REQUIRE(seg2.ToString() == bs2.Leading(19).ToString());

		if (shiftSize == 0)
		{
			REQUIRE(seg.ToString() == seg1.ToString());
		}
		else if (shiftSize >= seg.Size())
		{
			REQUIRE(seg.IsAllZeros());
		}
		else
		{
			CHECK(seg.Trailing(shiftSize).IsAllZeros());
			for (std::size_t i = 0; i + shiftSize < seg.Size(); ++i)
			{
				CHECK(seg[i + shiftSize] == seg1[i]);
			}
		}
	}

	SECTION("operator>>=")
	{
		DynamicBitset bs2 = bs0;
		auto seg = bs2.Segment(12, size - seg0.Size() - seg2.Size());
		seg >>= shiftSize;

		REQUIRE(seg0.ToString() == bs2.Trailing(12).ToString());
		REQUIRE(seg2.ToString() == bs2.Leading(19).ToString());

		if (shiftSize == 0)
		{
			REQUIRE(seg.ToString() == seg1.ToString());
		}
		else if (shiftSize >= seg.Size())
		{
			REQUIRE(seg.IsAllZeros());
		}
		else
		{
			CHECK(seg.Leading(shiftSize).IsAllZeros());
			for (std::size_t i = 0; i + shiftSize < seg.Size(); ++i)
			{
				CHECK(seg[i] == seg1[i + shiftSize]);
			}
		}
	}

	SECTION("operator<<")
	{
		DynamicBitset bs2 = bs0;
		const auto seg = bs2.Segment(12, size - seg0.Size() - seg2.Size());
		const auto shifted = seg << shiftSize;

		REQUIRE(shifted.Size() == seg.Size());
		REQUIRE(seg0.ToString() == bs2.Trailing(12).ToString());
		REQUIRE(seg2.ToString() == bs2.Leading(19).ToString());
		REQUIRE(seg1.ToString() == seg.ToString());

		if (shiftSize == 0)
		{
			REQUIRE(shifted.ToString() == seg1.ToString());
		}
		else if (shiftSize >= seg.Size())
		{
			REQUIRE(shifted.IsAllZeros());
		}
		else
		{
			CHECK(shifted.Trailing(shiftSize).IsAllZeros());
			for (std::size_t i = 0; i + shiftSize < seg.Size(); ++i)
			{
				CHECK(shifted[i + shiftSize] == seg1[i]);
			}
		}
	}

	SECTION("operator>>")
	{
		DynamicBitset bs2 = bs0;
		const auto seg = bs2.Segment(12, size - seg0.Size() - seg2.Size());
		const auto shifted = seg >> shiftSize;

		REQUIRE(shifted.Size() == seg.Size());
		REQUIRE(seg0.ToString() == bs2.Trailing(12).ToString());
		REQUIRE(seg2.ToString() == bs2.Leading(19).ToString());
		REQUIRE(seg1.ToString() == seg.ToString());

		if (shiftSize == 0)
		{
			REQUIRE(shifted.ToString() == seg1.ToString());
		}
		else if (shiftSize >= seg.Size())
		{
			REQUIRE(shifted.IsAllZeros());
		}
		else
		{
			CHECK(shifted.Leading(shiftSize).IsAllZeros());
			for (std::size_t i = 0; i + shiftSize < seg.Size(); ++i)
			{
				CHECK(shifted[i] == seg1[i + shiftSize]);
			}
		}
	}
}

TEST_CASE("IndexOfNextOne, IndexOfNextZero, IndexOfPreviousOne, and IndexOfPreviousZero")
{
	constexpr std::size_t size = 223;
	const auto seed = GENERATE(0, 1, 42, 69, 73, 420, 4242, 6969, 66872);

	const auto bitString0 = RandomBitString(size, seed);
	DynamicBitset bs(size);
	std::transform(bitString0.begin(), bitString0.end(), bs.begin(), [](const char c) { return c != '0'; });

	const auto seg = bs.Segment(62, 8);

	SECTION("IndexOfNextOne")
	{
		const unsigned pos = GENERATE(0, 8, 42, 63, 64, 65, 69, 73, 127, 128, 222);
		const auto expected = static_cast<std::size_t>(std::distance(
		        bs.begin(), std::find_if(bs.begin() + pos + 1, bs.end(), [](const bool b) { return b; })));
		const auto actual = bs.IndexOfNextOne(pos);
		CHECK(expected == actual);
	}

	SECTION("IndexOfNextZero")
	{
		const unsigned pos = GENERATE(0, 8, 42, 63, 64, 65, 69, 73, 127, 128, 222);
		const auto expected = static_cast<std::size_t>(std::distance(
		        bs.begin(), std::find_if(bs.begin() + pos + 1, bs.end(), [](const bool b) { return !b; })));
		const auto actual = bs.IndexOfNextZero(pos);
		CHECK(expected == actual);
	}

	SECTION("IndexOfPreviousOne")
	{
		const unsigned pos = GENERATE(8, 42, 63, 64, 65, 69, 73, 127, 128, 222);
		const auto actual = bs.IndexOfPreviousOne(pos);
		REQUIRE(bs[actual] == true);
		REQUIRE(!bs.Segment(actual + 1, pos - (actual + 1)).HasOnes());
	}

	SECTION("IndexOfPreviousZero")
	{
		const unsigned pos = GENERATE(8, 42, 63, 64, 65, 69, 73, 127, 128, 222);
		const auto actual = bs.IndexOfPreviousZero(pos);
		REQUIRE(bs[actual] == false);
		REQUIRE(!bs.Segment(actual + 1, pos - (actual + 1)).HasZeros());
	}

	SECTION("IndexOfNextOne w/ Segment")
	{
		const unsigned pos = GENERATE(0, 1, 2, 3, 4, 5, 6, 7);
		const auto expected = static_cast<std::size_t>(std::distance(
		        seg.begin(), std::find_if(seg.begin() + pos + 1, seg.end(), [](const bool b) { return b; })));
		const auto actual = seg.IndexOfNextOne(pos);

		CHECK(actual == expected);
	}

	SECTION("IndexOfNextZero w/ Segment")
	{
		const unsigned pos = GENERATE(0, 1, 2, 3, 4, 5, 6, 7);
		const auto expected = static_cast<std::size_t>(std::distance(
		        seg.begin(), std::find_if(seg.begin() + pos + 1, seg.end(), [](const bool b) { return !b; })));
		const auto actual = seg.IndexOfNextZero(pos);
		CHECK(actual == expected);
	}

	SECTION("IndexOfPreviousOne w/ Segment")
	{
		const unsigned pos = GENERATE(1, 2, 3, 4, 5, 6, 7);
		const auto actual = seg.IndexOfPreviousOne(pos);

		if (actual == seg.Size())
		{
			REQUIRE(!seg.Segment(0, pos - 1).HasOnes());
		}
		else
		{
			REQUIRE(seg[actual] == true);
			REQUIRE(!seg.Segment(actual + 1, pos - (actual + 1)).HasOnes());
		}
	}

	SECTION("IndexOfPreviousZero w/ Segment")
	{
		const unsigned pos = GENERATE(1, 2, 3, 4, 5, 6, 7);
		const auto actual = seg.IndexOfPreviousZero(pos);

		if (actual == seg.Size())
		{
			REQUIRE(!seg.Segment(0, pos - 1).HasZeros());
		}
		else
		{
			REQUIRE(seg[actual] == false);
			REQUIRE(!seg.Segment(actual + 1, pos - (actual + 1)).HasZeros());
		}
	}


	SECTION("Edge case")
	{
		CHECK(bs.IndexOfNextOne(222) == 223);
		CHECK(bs.IndexOfNextZero(222) == 223);
		CHECK(bs.IndexOfPreviousOne(0) == 223);
		CHECK(bs.IndexOfPreviousZero(0) == 223);

		bs.SetAll();
		CHECK(bs.IndexOfNextOne(10) == 11);
		CHECK(bs.IndexOfNextZero(10) == 223);
		CHECK(bs.IndexOfPreviousOne(10) == 9);
		CHECK(bs.IndexOfPreviousZero(10) == 223);

		bs.ResetAll();
		CHECK(bs.IndexOfNextOne(10) == 223);
		CHECK(bs.IndexOfNextZero(10) == 11);
		CHECK(bs.IndexOfPreviousOne(10) == 223);
		CHECK(bs.IndexOfPreviousZero(10) == 9);
	}
}

TEMPLATE_TEST_CASE("Cross assignment",
                   "[template]",
                   (Bitset<0>),
                   (Bitset<4>),
                   (Bitset<8>),
                   (Bitset<42>),
                   (Bitset<64>),
                   (Bitset<142>))
{
	const std::size_t seed = GENERATE(0, 1, 2, 3, 4, 5, 6, 7);
	const std::size_t size = TestType{}.Size();

	const auto bitString = RandomBitString(size, seed);

	TestType sbs{};
	std::transform(bitString.begin(), bitString.end(), std::begin(sbs), [](const char c) { return c != '0'; });

	{
		DynamicBitset dbs(size);
		dbs = sbs;
		CHECK(dbs.ToString() == sbs.ToString());

		dbs.FlipAll();
		if (size != 0)
		{
			CHECK(dbs.ToString() != sbs.ToString());
		}
		sbs = dbs;
		CHECK(dbs.ToString() == sbs.ToString());

		DynamicBitset dbs2{sbs};
		CHECK(dbs2.ToString() == sbs.ToString());

		DynamicBitset dbs3 = sbs;
		CHECK(dbs3.ToString() == sbs.ToString());
	}
	{
		DynamicBitset dbs(size * 2);

		dbs.Leading(size) = sbs;
		CHECK(dbs.Leading(size).ToString() == sbs.ToString());
		dbs.Trailing(size) = ~dbs.Leading(size);
		CHECK(dbs.Trailing(size).ToString() == (~sbs).ToString());

		dbs.Segment(size / 2, size) = sbs;
		CHECK(dbs.Leading(size / 2).ToString() == sbs.Leading(size / 2).ToString());
		CHECK(dbs.Trailing(size / 2).ToString() == (~sbs).Trailing(size / 2).ToString());
		CHECK(dbs.Segment(size / 2, size).ToString() == sbs.ToString());

		sbs = dbs.Segment(size / 3, size);
		CHECK(sbs.ToString() == dbs.Segment(size / 3, size).ToString());

		TestType sbs2{dbs.Segment(size / 3, size)};
		CHECK(sbs2.ToString() == dbs.Segment(size / 3, size).ToString());

		TestType sbs3 = dbs.Segment(size / 3, size);
		CHECK(sbs3.ToString() == dbs.Segment(size / 3, size).ToString());
	}

	{
		DynamicBitset dbs(size + 2);
		REQUIRE_THROWS([&] { dbs = sbs; }());
		REQUIRE_THROWS([&] { sbs = dbs; }());
	}
}
