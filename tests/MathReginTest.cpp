//
// Created by Andy on 4/7/2026.
//

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <SecUtility/Math/Regin.hpp>
#include <unordered_set>


using namespace SecUtility::Math;

#define DUAL_CHECK(...)                                                                                                \
	STATIC_CHECK(__VA_ARGS__);                                                                                         \
	CHECK(__VA_ARGS__)


TEMPLATE_TEST_CASE("Regin0 basic properties", "[template]", int, std::size_t, long, long long)
{
	SECTION("Diagonal elements")
	{
		// For diagonal elements (i, i), the formula is: i*(i+1)/2 + i = i*(i+3)/2
		DUAL_CHECK(Regin0<TestType>(0, 0) == 0);
		DUAL_CHECK(Regin0<TestType>(1, 1) == 2);
		DUAL_CHECK(Regin0<TestType>(2, 2) == 5);
		DUAL_CHECK(Regin0<TestType>(3, 3) == 9);
		DUAL_CHECK(Regin0<TestType>(4, 4) == 14);
		DUAL_CHECK(Regin0<TestType>(5, 5) == 20);
	}

	SECTION("Off diagonal elements")
	{
		// Upper triangular elements
		DUAL_CHECK(Regin0<TestType>(1, 0) == 1);
		DUAL_CHECK(Regin0<TestType>(2, 0) == 3);
		DUAL_CHECK(Regin0<TestType>(2, 1) == 4);
		DUAL_CHECK(Regin0<TestType>(3, 0) == 6);
		DUAL_CHECK(Regin0<TestType>(3, 1) == 7);
		DUAL_CHECK(Regin0<TestType>(3, 2) == 8);
	}

	SECTION("Symmetry property")
	{
		// Regin0(i, j) should equal Regin0(j, i)
		DUAL_CHECK(Regin0<TestType>(3, 1) == Regin0<TestType>(1, 3));
		DUAL_CHECK(Regin0<TestType>(5, 2) == Regin0<TestType>(2, 5));
		DUAL_CHECK(Regin0<TestType>(10, 7) == Regin0<TestType>(7, 10));
		DUAL_CHECK(Regin0<TestType>(0, 5) == Regin0<TestType>(5, 0));
	}

	SECTION("Monotonicity")
	{
		// For fixed i, Regin0(i, j) should increase with j
		for (TestType i = 0; i < 10; ++i)
		{
			for (TestType j = 0; j + 1 < i; ++j)
			{
				CHECK(Regin0<TestType>(i, j) < Regin0<TestType>(i, j + 1));
			}
		}
	}
}


TEMPLATE_TEST_CASE("Regin1 basic properties", "[template]", int, std::size_t, long, long long)
{
	SECTION("Diagonal elements")
	{
		// For 1-based indexing
		DUAL_CHECK(Regin1<TestType>(1, 1) == 1);
		DUAL_CHECK(Regin1<TestType>(2, 2) == 3);
		DUAL_CHECK(Regin1<TestType>(3, 3) == 6);
		DUAL_CHECK(Regin1<TestType>(4, 4) == 10);
		DUAL_CHECK(Regin1<TestType>(5, 5) == 15);
	}

	SECTION("Off diagonal elements")
	{
		DUAL_CHECK(Regin1<TestType>(2, 1) == 2);
		DUAL_CHECK(Regin1<TestType>(3, 1) == 4);
		DUAL_CHECK(Regin1<TestType>(3, 2) == 5);
		DUAL_CHECK(Regin1<TestType>(4, 1) == 7);
		DUAL_CHECK(Regin1<TestType>(4, 2) == 8);
		DUAL_CHECK(Regin1<TestType>(4, 3) == 9);
	}

	SECTION("Symmetry property")
	{
		// Regin1(i, j) should equal Regin1(j, i)
		DUAL_CHECK(Regin1<TestType>(3, 1) == Regin1<TestType>(1, 3));
		DUAL_CHECK(Regin1<TestType>(5, 2) == Regin1<TestType>(2, 5));
		DUAL_CHECK(Regin1<TestType>(10, 7) == Regin1<TestType>(7, 10));
	}

	SECTION("Relationship to Regin0")
	{
		// Regin1(i, j) should equal Regin0(i-1, j-1) + 1
		DUAL_CHECK(Regin1<TestType>(1, 1) == Regin0<TestType>(0, 0) + 1);
		DUAL_CHECK(Regin1<TestType>(3, 2) == Regin0<TestType>(2, 1) + 1);
		DUAL_CHECK(Regin1<TestType>(5, 4) == Regin0<TestType>(4, 3) + 1);
		DUAL_CHECK(Regin1<TestType>(10, 1) == Regin0<TestType>(9, 0) + 1);
	}
}


TEMPLATE_TEST_CASE("Regin0_4D", "[template]", int, std::size_t, long, long long)
{
	SECTION("Basic_4D_indexing")
	{
		// Regin0(i, j, k, l) = Regin0(Regin0(i, j), Regin0(k, l))
		// This maps 4D symmetric tensor indices to 1D

		DUAL_CHECK(Regin0<TestType>(0, 0, 0, 0) == 0);
		DUAL_CHECK(Regin0<TestType>(1, 0, 0, 0) == 1);
		DUAL_CHECK(Regin0<TestType>(0, 1, 0, 0) == 1);  // Symmetric in first pair
		DUAL_CHECK(Regin0<TestType>(0, 0, 1, 0) == 1);  // Symmetric in second pair
		DUAL_CHECK(Regin0<TestType>(0, 0, 0, 1) == 1);  // Symmetric in second pair

		DUAL_CHECK(Regin0<TestType>(1, 1, 0, 0) == 3);
		DUAL_CHECK(Regin0<TestType>(0, 0, 1, 1) == 3);
		DUAL_CHECK(Regin0<TestType>(0, 1, 0, 1) == 2);
	}

	SECTION("Symmetry across all indices")
	{
		// All permutations of (2, 1, 1, 0) should give the same result
		const TestType expected = Regin0<TestType>(2, 1, 1, 0);
		DUAL_CHECK(Regin0<TestType>(2, 1, 0, 1) == expected);
		DUAL_CHECK(Regin0<TestType>(1, 2, 1, 0) == expected);
		DUAL_CHECK(Regin0<TestType>(1, 2, 0, 1) == expected);
		DUAL_CHECK(Regin0<TestType>(1, 0, 2, 1) == expected);
		DUAL_CHECK(Regin0<TestType>(1, 0, 1, 2) == expected);
		DUAL_CHECK(Regin0<TestType>(0, 1, 2, 1) == expected);
		DUAL_CHECK(Regin0<TestType>(0, 1, 1, 2) == expected);
	}

	SECTION("Known values for small indices")
	{
		DUAL_CHECK(Regin0<TestType>(0, 0, 0, 0) == 0);
		DUAL_CHECK(Regin0<TestType>(1, 1, 1, 1) == 5);
		DUAL_CHECK(Regin0<TestType>(2, 2, 2, 2) == 20);
		DUAL_CHECK(Regin0<TestType>(3, 3, 3, 3) == 54);
	}
}


TEMPLATE_TEST_CASE("Regin1_4D", "[template]", int, std::size_t, long, long long)
{
	SECTION("Basic_4D_indexing")
	{
		DUAL_CHECK(Regin1<TestType>(1, 1, 1, 1) == 1);
		DUAL_CHECK(Regin1<TestType>(2, 1, 1, 1) == 2);
		DUAL_CHECK(Regin1<TestType>(1, 2, 1, 1) == 2);
		DUAL_CHECK(Regin1<TestType>(1, 1, 2, 1) == 2);
		DUAL_CHECK(Regin1<TestType>(1, 1, 1, 2) == 2);
	}

	SECTION("Relationship to Regin0")
	{
		DUAL_CHECK(Regin1<TestType>(1, 1, 1, 1) == Regin0<TestType>(0, 0, 0, 0) + 1);
		DUAL_CHECK(Regin1<TestType>(2, 1, 1, 1) == Regin0<TestType>(1, 0, 0, 0) + 1);
		DUAL_CHECK(Regin1<TestType>(3, 2, 2, 1) == Regin0<TestType>(2, 1, 1, 0) + 1);
	}

	SECTION("Symmetry across all indices")
	{
		const TestType expected = Regin1<TestType>(3, 2, 2, 1);
		DUAL_CHECK(Regin1<TestType>(3, 2, 1, 2) == expected);
		DUAL_CHECK(Regin1<TestType>(2, 3, 2, 1) == expected);
		DUAL_CHECK(Regin1<TestType>(2, 3, 1, 2) == expected);
	}
}


TEMPLATE_TEST_CASE("Regin0 power of 2", "[template]", int, std::size_t, long, long long)
{
	SECTION("N_equals_2")
	{
		// Regin0<2>(i) = Regin0(i, i)
		DUAL_CHECK(Regin0<2, TestType>(0) == 0);
		DUAL_CHECK(Regin0<2, TestType>(1) == 2);
		DUAL_CHECK(Regin0<2, TestType>(2) == 5);
		DUAL_CHECK(Regin0<2, TestType>(3) == 9);
		DUAL_CHECK(Regin0<2, TestType>(5) == 20);
		DUAL_CHECK(Regin0<2, TestType>(10) == 65);
	}

	SECTION("N_equals_4")
	{
		// Regin0<4>(i) = Regin0(Regin0(i, i), Regin0(i, i))
		DUAL_CHECK(Regin0<4, TestType>(0) == 0);
		DUAL_CHECK(Regin0<4, TestType>(1) == 5);
		DUAL_CHECK(Regin0<4, TestType>(2) == 20);
		DUAL_CHECK(Regin0<4, TestType>(3) == 54);

		// Verify against explicit calculation
		// For i=2: Regin0(2, 2) = 3, Regin0(3, 3) = 6
		DUAL_CHECK(Regin0<4, TestType>(2) == Regin0<TestType>(Regin0<TestType>(2, 2), Regin0<TestType>(2, 2)));
	}

	SECTION("N_equals_8")
	{
		DUAL_CHECK(Regin0<8, TestType>(0) == 0);
		DUAL_CHECK(Regin0<8, TestType>(1) == 20);
		DUAL_CHECK(Regin0<8, TestType>(2) == 230);
	}

	SECTION("N_equals_16")
	{
		DUAL_CHECK(Regin0<16, TestType>(0) == 0);
		DUAL_CHECK(Regin0<16, TestType>(1) == 230);
		DUAL_CHECK(Regin0<16, TestType>(2) == 26795);
	}
}


TEMPLATE_TEST_CASE("Regin1 power of 2", "[template]", int, std::size_t, long, long long)
{
	SECTION("N_equals_2")
	{
		// Regin1<2>(i) = Regin1(i, i)
		DUAL_CHECK(Regin1<2, TestType>(1) == 1);
		DUAL_CHECK(Regin1<2, TestType>(2) == 3);
		DUAL_CHECK(Regin1<2, TestType>(3) == 6);
		DUAL_CHECK(Regin1<2, TestType>(4) == 10);
		DUAL_CHECK(Regin1<2, TestType>(5) == 15);
		DUAL_CHECK(Regin1<2, TestType>(6) == 21);
	}

	SECTION("N_equals_4")
	{
		DUAL_CHECK(Regin1<4, TestType>(1) == 1);
		DUAL_CHECK(Regin1<4, TestType>(2) == 6);
		DUAL_CHECK(Regin1<4, TestType>(3) == 21);
		DUAL_CHECK(Regin1<4, TestType>(4) == 55);

		// Verify against explicit calculation
		DUAL_CHECK(Regin1<4, TestType>(3) == Regin1<TestType>(Regin1<TestType>(3, 3), Regin1<TestType>(3, 3)));
	}

	SECTION("Relationship to Regin0")
	{
		// Regin1<N>(i) = Regin0<N>(i-1) + 1
		DUAL_CHECK(Regin1<2, TestType>(5) == Regin0<2, TestType>(4) + 1);
		DUAL_CHECK(Regin1<4, TestType>(3) == Regin0<4, TestType>(2) + 1);
		DUAL_CHECK(Regin1<8, TestType>(2) == Regin0<8, TestType>(1) + 1);
	}
}


TEMPLATE_TEST_CASE("Regin injectivity", "[template]", int, long)
{
	SECTION("Regin0 is injective for valid pairs")
	{
		// Each unique (i, j) pair should map to a unique index
		// for i >= j >= 0 within a reasonable range

		std::unordered_set<TestType> indices;
		const TestType max = 20;

		for (TestType i = 0; i <= max; ++i)
		{
			for (TestType j = 0; j <= i; ++j)
			{
				TestType idx = Regin0<TestType>(i, j);
				CHECK(indices.insert(idx).second);  // Should not have duplicates
			}
		}
	}

	SECTION("Regin1 is injective for valid pairs")
	{
		std::unordered_set<TestType> indices;
		const TestType max = 20;

		for (TestType i = 1; i <= max; ++i)
		{
			for (TestType j = 1; j <= i; ++j)
			{
				TestType idx = Regin1<TestType>(i, j);
				CHECK(indices.insert(idx).second);  // Should not have duplicates
			}
		}
	}
}

TEST_CASE("Regin triangular matrix storage")
{
	SECTION("Storage size for nxn matrix")
	{
		// For an n×n symmetric matrix, we need n*(n+1)/2 elements
		// The last element at (n-1, n-1) in 0-based indexing should be at index n*(n+1)/2 - 1

		for (int n = 1; n <= 20; ++n)
		{
			const auto lastIndex0 = Regin0<int>(n - 1, n - 1);
			const auto expectedSize = n * (n + 1) / 2;
			CHECK(lastIndex0 + 1 == expectedSize);

			const auto lastIndex1 = Regin1<int>(n, n);
			CHECK(lastIndex1 == expectedSize);
		}
	}

	SECTION("Matrix index mapping")
	{
		// Simulate storing a 3x3 symmetric matrix
		constexpr int n = 3;
		std::array<double, n*(n + 1) / 2> storage{};

		// Fill with some values
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j <= i; ++j)
			{
				storage[Regin0(i, j)] = static_cast<double>(i * 10 + j);
			}
		}

		// Verify we can retrieve values using symmetric access
		CHECK(storage[Regin0(0, 0)] == 0);
		CHECK(storage[Regin0(1, 0)] == 10);
		CHECK(storage[Regin0(0, 1)] == 10);  // Same as (1, 0)
		CHECK(storage[Regin0(1, 1)] == 11);
		CHECK(storage[Regin0(2, 0)] == 20);
		CHECK(storage[Regin0(2, 1)] == 21);
		CHECK(storage[Regin0(2, 2)] == 22);
	}
}


TEST_CASE("Regin edge cases")
{
	SECTION("Zero indices")
	{
		DUAL_CHECK(Regin0(0, 0) == 0);
		DUAL_CHECK(Regin1(1, 1) == 1);
		DUAL_CHECK(Regin0(0, 0, 0, 0) == 0);
		DUAL_CHECK(Regin1(1, 1, 1, 1) == 1);
		DUAL_CHECK(Regin0<2>(0) == 0);
		DUAL_CHECK(Regin1<2>(1) == 1);
	}

	SECTION("Small matrices")
	{
		// 1x1 matrix
		DUAL_CHECK(Regin0(0, 0) == 0);
		DUAL_CHECK(Regin1(1, 1) == 1);

		// 2x2 matrix
		DUAL_CHECK(Regin0(0, 0) == 0);
		DUAL_CHECK(Regin0(1, 0) == 1);
		DUAL_CHECK(Regin0(1, 1) == 2);

		DUAL_CHECK(Regin1(1, 1) == 1);
		DUAL_CHECK(Regin1(2, 1) == 2);
		DUAL_CHECK(Regin1(2, 2) == 3);
	}

	SECTION("Large indices")
	{
		// Test with larger indices to ensure no overflow issues
		constexpr std::size_t large = 1000;
		constexpr auto idx = Regin0<std::size_t>(large, large / 2);

		// The formula is: large * (large + 1) / 2 + large / 2
		constexpr auto expected = large * (large + 1) / 2 + large / 2;
		DUAL_CHECK(idx == expected);
	}
}


TEMPLATE_TEST_CASE("Regin type consistency", "[template]", int, std::size_t, long, long long)
{
	SECTION("Same values for different types")
	{
		// For small values, different integer types should give the same results
		DUAL_CHECK(Regin0<TestType>(5, 3) == Regin0<int>(5, 3));
		DUAL_CHECK(Regin1<TestType>(5, 3) == Regin1<int>(5, 3));
		DUAL_CHECK(Regin0<TestType>(3, 2, 1, 0) == Regin0<int>(3, 2, 1, 0));
		DUAL_CHECK(Regin1<TestType>(4, 3, 2, 1) == Regin1<int>(4, 3, 2, 1));
		DUAL_CHECK(Regin0<4, TestType>(3) == Regin0<4, int>(3));
		DUAL_CHECK(Regin1<8, TestType>(2) == Regin1<8, int>(2));
	}
}
