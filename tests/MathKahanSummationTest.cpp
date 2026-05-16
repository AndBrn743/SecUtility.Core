//
// Created by Andy on 5/16/2026.
//

#include <SecUtility/Math/KahanSummation.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace
{
	template <typename Accumulator>
	double Accumulate(const std::vector<double>& values)
	{
		Accumulator accumulator;

		for (double value : values)
		{
			accumulator.AddTerm(value);
		}

		return accumulator.Sum();
	}

	double NaiveSum(const std::vector<double>& values)
	{
		return std::reduce(values.begin(), values.end());
	}
}

using namespace SecUtility::Math;
using Catch::Approx;

TEMPLATE_TEST_CASE("Kahan-like accumulator sums simple values correctly",
                   "[template]",
                   (KahanAccumulator<double>),
                   (KahanBabushkaNeumaierAccumulator<double>),
                   (KahanBabushkaKleinAccumulator<double>))
{
	TestType accumulator;
	accumulator.AddTerm(1.0).AddTerms(2.0, 3.0);
	REQUIRE(accumulator.Sum() == 6.0);
}

TEMPLATE_TEST_CASE("Compensated summation handles cancellation",
                   "[template]",
                   (KahanAccumulator<double>),
                   (KahanBabushkaNeumaierAccumulator<double>),
                   (KahanBabushkaKleinAccumulator<double>))
{
	const std::vector<double> values = {10000.0, 3.14159, 2.71828};
	const double naive = NaiveSum(values);
	const double kahan = Accumulate<TestType>(values);  // passed
	constexpr auto reference = 10005.85987;
	REQUIRE(naive != reference);
	REQUIRE(Abs(kahan - reference) < Abs(naive - reference));
	if (!std::is_same_v<KahanAccumulator<double>, TestType>)
	{
		REQUIRE(kahan == reference);
	}
}

TEMPLATE_TEST_CASE("Compensated summation handles catastrophic cancellation",
                   "[template]",
                   // (KahanAccumulator<double>),  // <- this one will fail
                   (KahanBabushkaNeumaierAccumulator<double>),
                   (KahanBabushkaKleinAccumulator<double>))
{
	// Exact mathematical result:
	//
	// (1e16 + 1 - 1e16) = 1
	//
	// Naive summation usually becomes 0 because
	// the +1 disappears.

	const std::vector<double> values = {1e16, 1.0, -1e16};

	const double naive = NaiveSum(values);

	const double kahan = Accumulate<TestType>(values);
	REQUIRE(naive == 0);
	REQUIRE(kahan == 1);
}

TEMPLATE_TEST_CASE("Compensated summation preserves many tiny increments",
                   "[template]",
                   (KahanAccumulator<double>),
                   (KahanBabushkaNeumaierAccumulator<double>),
                   (KahanBabushkaKleinAccumulator<double>))
{
	std::vector<double> values;

	values.reserve(1'000'001);

	values.push_back(1.0);

	for (size_t i = 0; i < 1'000'000; ++i)
	{
		values.push_back(1e-16);
	}

	const double exact = 1.0 + 1'000'000.0 * 1e-16;

	const double naive = NaiveSum(values);

	const double kahan = Accumulate<TestType>(values);

	REQUIRE(std::abs(kahan - exact) < std::abs(naive - exact));
}

TEMPLATE_TEST_CASE("Positive and negative values cancel correctly",
                   "[template]",
                   (KahanAccumulator<double>),
                   (KahanBabushkaNeumaierAccumulator<double>),
                   (KahanBabushkaKleinAccumulator<double>))
{
	std::vector<double> values;

	for (int i = 0; i < 100000; ++i)
	{
		values.push_back(1e-8);
		values.push_back(-1e-8);
	}

	const double kahan = Accumulate<TestType>(values);

	REQUIRE(kahan == Approx(0.0).margin(1e-20));
}

TEMPLATE_TEST_CASE("Order sensitivity is reduced",
                   "[template]",
                   (KahanAccumulator<double>),
                   (KahanBabushkaNeumaierAccumulator<double>),
                   (KahanBabushkaKleinAccumulator<double>))
{
	std::vector<double> ascending;
	std::vector<double> descending;

	for (int i = 0; i < 100000; ++i)
	{
		ascending.push_back(1e-10);
	}

	ascending.push_back(1.0);

	descending = ascending;

	reverse(descending.begin(), descending.end());

	const double naiveAscending = NaiveSum(ascending);

	const double naiveDescending = NaiveSum(descending);

	const double kahanAscending = Accumulate<TestType>(ascending);

	const double kahanDescending = Accumulate<TestType>(descending);

	REQUIRE(naiveAscending != naiveDescending);

	REQUIRE(kahanAscending == Approx(kahanDescending).epsilon(1e-14));
}

TEST_CASE("KBN is at least as accurate as Kahan")
{
	std::vector<double> values;

	values.push_back(1e16);

	for (int i = 0; i < 100000; ++i)
	{
		values.push_back(1e-8);
	}

	values.push_back(-1e16);

	const double exact = 100000.0 * 1e-8;

	const double kahan = Accumulate<KahanAccumulator<double>>(values);

	const double kbn = Accumulate<KahanBabushkaNeumaierAccumulator<double>>(values);

	REQUIRE(std::abs(kbn - exact) <= std::abs(kahan - exact));
}

TEST_CASE("KBK is at least as accurate as KBN")
{
	std::vector<double> values;

	std::mt19937 rng(1337);

	std::uniform_real_distribution<double> distribution(-1e10, 1e10);

	for (size_t i = 0; i < 100000; ++i)
	{
		values.push_back(distribution(rng));
	}

	// Long double reference.
	long double reference = 0;

	for (double value : values)
	{
		reference += static_cast<long double>(value);
	}

	const double kbn = Accumulate<KahanBabushkaNeumaierAccumulator<double>>(values);

	const double kbk = Accumulate<KahanBabushkaKleinAccumulator<double>>(values);

	const long double exact = reference;

	REQUIRE(std::abs(static_cast<long double>(kbk) - exact) <= std::abs(static_cast<long double>(kbn) - exact));
}

TEMPLATE_TEST_CASE("Empty accumulator returns zero",
                   "[template]",
                   (KahanAccumulator<double>),
                   (KahanBabushkaNeumaierAccumulator<double>),
                   (KahanBabushkaKleinAccumulator<double>))
{
	TestType accumulator;
	REQUIRE(accumulator.Sum() == 0.0);
}


TEST_CASE("Compensated summation's range adapter works")
{
	const std::vector<double> values = {10000.0, 3.14159, 2.71828};

	SECTION("Sum with identity projection")
	{
		const double naive = NaiveSum(values);
		const double kahan = KahanBabushkaNeumaierSum(values);
		constexpr auto reference = 10005.85987;
		REQUIRE(naive != reference);
		REQUIRE(Abs(kahan - reference) < Abs(naive - reference));
		REQUIRE(kahan == reference);
	}

	SECTION("Sum with non-identity projection")
	{
		const double naive = std::reduce(values.begin(),
		                                 values.end(),
		                                 double{0},
		                                 [](const auto acc, const auto value) { return acc - value; });
		const double kahan = KahanBabushkaNeumaierSum(values, std::negate<>{});
		constexpr auto reference = -10005.85987;
		REQUIRE(naive != reference);
		REQUIRE(Abs(kahan - reference) < Abs(naive - reference));
		REQUIRE(kahan == reference);
	}
}
