//
// Created by Claude on 2/9/2026.
//

#include <SecUtility/Misc/Random.hpp>

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <set>
#include <string>

using namespace SecUtility;


TEST_CASE("Random - Singleton behavior")
{
	SECTION("Get returns same instance")
	{
		auto& r1 = Random::Get();
		auto& r2 = Random::Get();
		CHECK(&r1 == &r2);
	}

	SECTION("SetSeed affects subsequent calls")
	{
		Random::SetSeed(42);
		const auto v1 = Random::Next();

		Random::SetSeed(42);
		const auto v2 = Random::Next();

		CHECK(v1 == v2);
	}
}


TEST_CASE("Random - Seeding produces reproducible sequences")
{
	Random::SetSeed(12345);

	std::array<Int32, 10> firstRun{};
	for (int& i : firstRun)
	{
		i = Random::Next();
	}

	// Reset with same seed
	Random::SetSeed(12345);

	std::array<Int32, 10> secondRun = {};
	for (int& i : secondRun)
	{
		i = Random::Next();
	}

	CHECK(firstRun == secondRun);
}


TEST_CASE("Random - Different seeds produce different sequences")
{
	Random::SetSeed(111);
	const auto v1 = Random::Next();

	Random::SetSeed(222);
	const auto v2 = Random::Next();

	Random::SetSeed(111);
	const auto v3 = Random::Next();

	CHECK(v1 != v2);
	CHECK(v1 == v3);
}


TEST_CASE("Random - Range constraints (integral types)")
{
	SECTION("Int32 within range")
	{
		for (int i = 0; i < 100; ++i)
		{
			constexpr Int32 min = 100;
			constexpr Int32 max = 200;

			Int32 value = Random::NextInt32(min, max);
			CHECK(value >= min);
			CHECK(value <= max);
		}
	}

	SECTION("Int64 within range")
	{
		for (int i = 0; i < 50; ++i)
		{
			constexpr Int64 min = 1000LL;
			constexpr Int64 max = 2000LL;

			Int64 value = Random::NextInt64(min, max);
			CHECK(value >= min);
			CHECK(value <= max);
		}
	}

	SECTION("UInt16 within range")
	{
		for (int i = 0; i < 100; ++i)
		{
			constexpr UInt16 min = 10;
			constexpr UInt16 max = 100;

			UInt16 value = Random::NextUInt16(min, max);
			CHECK(value >= min);
			CHECK(value <= max);
		}
	}

	SECTION("UInt32 full range")
	{
		CHECK_NOTHROW(Random::NextUInt32());
	}
}


TEST_CASE("Random - Range constraints (floating point types)")
{
	SECTION("Double within range")
	{
		for (int i = 0; i < 100; ++i)
		{
			constexpr double min = 10.0;
			constexpr double max = 20.0;

			double value = Random::NextDouble(min, max);
			CHECK(value >= min);
			CHECK(value <= max);
		}
	}

	SECTION("Single/Float within range")
	{
		for (int i = 0; i < 100; ++i)
		{
			constexpr float min = 5.0f;
			constexpr float max = 15.0f;

			float value = Random::NextSingle(min, max);
			CHECK(value >= min);
			CHECK(value <= max);
		}
	}

	SECTION("Double default range [0, 1]")
	{
		for (int i = 0; i < 100; ++i)
		{
			double value = Random::NextDouble();
			CHECK(value >= 0.0);
			CHECK(value <= 1.0);
		}
	}
}


TEST_CASE("Random - Generates variety of values")
{
	SECTION("Integer variety")
	{
		std::set<Int32> values;
		constexpr int iterations = 1000;

		for (int i = 0; i < iterations; ++i)
		{
			values.insert(Random::NextInt32(0, 100));
		}

		// With 100 possible values and 1000 draws, we should get many unique values
		CHECK(values.size() > 50);
	}

	SECTION("Floating point variety")
	{
		std::set<double> values;
		constexpr int iterations = 100;

		for (int i = 0; i < iterations; ++i)
		{
			values.insert(Random::NextDouble(0.0, 1.0));
		}

		// Floating point should give many unique values
		CHECK(values.size() > 90);
	}
}


TEST_CASE("Random - NextString length")
{
	SECTION("Default length")
	{
		const std::string s = Random::NextString();
		CHECK(s.length() == 16);
	}

	SECTION("Custom length")
	{
		for (int len = 1; len <= 100; len += 10)
		{
			const std::string s = Random::NextString(len);
			CHECK(s.length() == static_cast<size_t>(len));
		}
	}

	SECTION("Zero length")
	{
		std::string s = Random::NextString(0);
		CHECK(s.empty());
	}
}


TEST_CASE("Random - NextString character set")
{
	SECTION("Uses only characters from provided set")
	{
		const std::string charset = "ABC";
		constexpr int iterations = 100;

		for (int i = 0; i < iterations; ++i)
		{
			const std::string s = Random::NextString(50, charset); 
			for (char c : s)
			{
				CHECK(charset.find(c) != std::string::npos);
			}
		}
	}

	SECTION("Default character set")
	{
		const std::string defaultCharset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890";

		const std::string s = Random::NextString(100); 
		for (char c : s)
		{
			CHECK(defaultCharset.find(c) != std::string::npos);
		}
	}

	SECTION("Single character charset")
	{
		std::string s = Random::NextString(20, "X");
		CHECK(s == std::string(20, 'X'));
	}
}


TEST_CASE("Random - NextString variety")
{
	SECTION("Generates different strings")
	{
		std::set<std::string> strings;

		for (int i = 0; i < 100; ++i)
		{
			strings.insert(Random::NextString(10, "ABCD"));
		}

		// Should get variety, not all the same string
		CHECK(strings.size() > 10);
	}
}


TEST_CASE("Random - Edge cases")
{
	SECTION("Min equals max")
	{
		CHECK(Random::NextInt32(42, 42) == 42);
		CHECK(Random::NextInt64(100LL, 100LL) == 100LL);
		CHECK(Random::NextDouble(3.14, 3.14) == 3.14);
	}

	SECTION("Small range")
	{
		std::set<Int32> values;
		for (int i = 0; i < 100; ++i)
		{
			values.insert(Random::NextInt32(0, 1));
		}

		// Should get both 0 and 1
		CHECK(values.size() >= 2);
		CHECK(values.count(0) > 0);
		CHECK(values.count(1) > 0);
	}
}


TEST_CASE("Random - Basic distribution sanity check")
{
	// This is a very basic sanity check, not a rigorous statistical test
	SECTION("Uniform-ish distribution")
	{
		constexpr Int32 min = 0;
		constexpr Int32 max = 10;
		constexpr int samplesPerBucket = 100;
		constexpr int totalSamples = (max - min + 1) * samplesPerBucket;

		std::array<int, 11> counts{};  // counts[0] to counts[10]

		for (int i = 0; i < totalSamples; ++i)
		{
			const Int32 value = Random::NextInt32(min, max);
			counts[value]++;
		}

		// Each bucket should have roughly samplesPerBucket counts
		// Allow significant tolerance since it's random
		constexpr int minExpected = samplesPerBucket / 2;
		constexpr int maxExpected = samplesPerBucket * 2;

		for (const int count : counts)
		{
			CHECK(count >= minExpected);
			CHECK(count <= maxExpected);
		}
	}
}
