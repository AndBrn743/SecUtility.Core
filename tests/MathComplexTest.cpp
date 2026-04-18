// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Math/Complex.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <limits>

TEMPLATE_TEST_CASE("Complex arithmetic with integers", "[template]", float, double, (long double), short, int, long)
{
	using Complex = std::complex<TestType>;

	SECTION("complex * integer")
	{
		const Complex z{3, 4};

		// complex * int
		const auto result1 = z * 2;
		CHECK(result1.real() == TestType{6});
		CHECK(result1.imag() == TestType{8});

		// int * complex
		const auto result2 = 2 * z;
		CHECK(result2.real() == TestType{6});
		CHECK(result2.imag() == TestType{8});

		// complex * long long
		const auto result3 = z * 3LL;
		CHECK(result3.real() == TestType{9});
		CHECK(result3.imag() == TestType{12});

		// Test with negative integer
		const auto result4 = z * (-2);
		CHECK(result4.real() == TestType{-6});
		CHECK(result4.imag() == TestType{-8});

		// Test with zero
		const auto result5 = z * 0;
		CHECK(result5.real() == TestType{0});
		CHECK(result5.imag() == TestType{0});
	}

	SECTION("complex + integer")
	{
		const Complex z{3, 4};

		// complex + int
		const auto result1 = z + 2;
		CHECK(result1.real() == TestType{5});
		CHECK(result1.imag() == TestType{4});

		// int + complex (commutative)
		const auto result2 = 2 + z;
		CHECK(result2.real() == TestType{5});
		CHECK(result2.imag() == TestType{4});

		// Test with negative integer
		const auto result3 = z + (-1);
		CHECK(result3.real() == TestType{2});
		CHECK(result3.imag() == TestType{4});

		// Test with zero
		const auto result4 = z + 0;
		CHECK(result4.real() == TestType{3});
		CHECK(result4.imag() == TestType{4});
	}

	SECTION("complex - integer")
	{
		const Complex z{5, 7};

		// complex - int
		const auto result1 = z - 2;
		CHECK(result1.real() == TestType{3});
		CHECK(result1.imag() == TestType{7});

		// Test with negative integer
		const auto result2 = z - (-1);
		CHECK(result2.real() == TestType{6});
		CHECK(result2.imag() == TestType{7});

		// Test with zero
		const auto result3 = z - 0;
		CHECK(result3.real() == TestType{5});
		CHECK(result3.imag() == TestType{7});
	}

	SECTION("integer - complex (critical bug test)")
	{
		const Complex z{3, 4};

		// int - complex: 5 - (3 + 4i) = 2 - 4i
		const auto result1 = 5 - z;
		CHECK(result1.real() == TestType{2});
		CHECK(result1.imag() == TestType{-4});

		// Test with different values: 10 - (2 + 3i) = 8 - 3i
		const Complex z2{2, 3};
		const auto result2 = 10 - z2;
		CHECK(result2.real() == TestType{8});
		CHECK(result2.imag() == TestType{-3});

		// Test with negative result: 2 - (5 + 1i) = -3 - 1i
		const Complex z3{5, 1};
		const auto result3 = 2 - z3;
		CHECK(result3.real() == TestType{-3});
		CHECK(result3.imag() == TestType{-1});

		// Test with zero imaginary: 7 - (3 + 0i) = 4 + 0i
		const Complex z4{3, 0};
		const auto result4 = 7 - z4;
		CHECK(result4.real() == TestType{4});
		CHECK(result4.imag() == TestType{0});

		// Test with zero real: 5 - (0 + 2i) = 5 - 2i
		const Complex z5{0, 2};
		const auto result5 = 5 - z5;
		CHECK(result5.real() == TestType{5});
		CHECK(result5.imag() == TestType{-2});
	}

	SECTION("complex / integer")
	{
		const Complex z{6, 8};

		// complex / int
		const auto result1 = z / 2;
		CHECK(result1.real() == TestType{3});
		CHECK(result1.imag() == TestType{4});

		// Test with odd numbers (should work with floating point)
		const Complex z2{5, 7};
		const auto result2 = z2 / 2;
		CHECK(result2.real() == static_cast<TestType>(2.5));
		CHECK(result2.imag() == static_cast<TestType>(3.5));

		// Test with negative divisor
		const auto result3 = z / (-2);
		CHECK(result3.real() == TestType{-3});
		CHECK(result3.imag() == TestType{-4});
	}

	SECTION("integer / complex")
	{
		const Complex z{2, 2};

		// int / complex: 8 / (2 + 2i) = 8 * (2 - 2i) / ((2+2i)(2-2i)) = 8 * (2-2i) / 8 = 2 - 2i
		const auto result1 = 8 / z;
		CHECK(result1.real() == TestType{2});
		CHECK(result1.imag() == TestType{-2});

		// Test with pure imaginary: 4 / (0 + 2i) = -2i
		const Complex z2{0, 2};
		const auto result2 = 4 / z2;
		CHECK(result2.real() == TestType{0});
		CHECK(result2.imag() == TestType{-2});

		// Test with pure real: 10 / (5 + 0i) = 2
		const Complex z3{5, 0};
		const auto result3 = 10 / z3;
		CHECK(result3.real() == TestType{2});
		CHECK(result3.imag() == TestType{0});
	}

	SECTION("Mixed type operations")
	{
		const Complex z{3, 4};

		// Test promotion with different integer types
		const auto result1 = z * static_cast<long>(2);
		CHECK(result1.real() == TestType{6});
		CHECK(result1.imag() == TestType{8});

		const auto result2 = z + static_cast<short>(1);
		CHECK(result2.real() == TestType{4});
		CHECK(result2.imag() == TestType{4});

		const auto result3 = static_cast<long long>(10) - z;
		CHECK(result3.real() == TestType{7});
		CHECK(result3.imag() == TestType{-4});
	}

	SECTION("Chained operations")
	{
		const Complex z{1, 1};

		// (z * 2) + 3 = (2 + 2i) + 3 = 5 + 2i
		const auto result1 = z * 2 + 3;
		CHECK(result1.real() == TestType{5});
		CHECK(result1.imag() == TestType{2});

		// (z + 1) * 2 = (2 + 1i) * 2 = 4 + 2i
		const auto result2 = (z + 1) * 2;
		CHECK(result2.real() == TestType{4});
		CHECK(result2.imag() == TestType{2});

		// 10 - (z * 2) = 10 - (2 + 2i) = 8 - 2i
		const auto result3 = 10 - z * 2;
		CHECK(result3.real() == TestType{8});
		CHECK(result3.imag() == TestType{-2});
	}

	SECTION("Type promotion")
	{
		const Complex z{3, 4};

		// Verify common_type is used correctly
		const auto result1 = z * 2;
		static_assert(std::is_same_v<std::decay_t<decltype(result1)>, std::complex<std::common_type_t<TestType, int>>>);

		// With different integer types
		const auto result2 = z * 2L;
		static_assert(std::is_same_v<std::decay_t<decltype(result2)>, std::complex<std::common_type_t<TestType, long>>>);
	}
}
