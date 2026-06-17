// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <SecUtility/Math/Search.hpp>

#include <functional>


using namespace SecUtility;
using namespace SecUtility::Math;
using Catch::Approx;


//----------------------------------------------------------------------------------------------------------------------
// Free functions used as function-pointer targets
//----------------------------------------------------------------------------------------------------------------------

namespace
{
	[[maybe_unused]] double SquareMinusTwo(const double x)
	{
		return x * x - 2.0;
	}

	[[maybe_unused]] float SquareMinusTwoF(const float x)
	{
		return x * x - 2.0F;
	}
}


//----------------------------------------------------------------------------------------------------------------------
// Bracketing & convergence on textbook roots
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("BisectionSearch finds sqrt(2)", "[Math][Search]")
{
	const auto result = BisectionSearch(&SquareMinusTwo, Interval<double>{1.0, 2.0});

	REQUIRE(result.IsConverged);
	CHECK(result.Position == Approx(1.4142135623730951).margin(1e-12));
	CHECK(result.FunctionValue == Approx(0.0).margin(1e-9));
	CHECK(result.IterationCount > 0);
}

TEST_CASE("BisectionSearch finds a negative root", "[Math][Search]")
{
	const auto result = BisectionSearch([](const double x) { return x * x - 4.0; }, Interval<double>{-3.0, 0.0});

	REQUIRE(result.IsConverged);
	CHECK(result.Position == Approx(-2.0).margin(1e-12));
	CHECK(result.FunctionValue == Approx(0.0).margin(1e-9));
}

TEST_CASE("BisectionSearch finds the root of a strictly monotone cubic", "[Math][Search]")
{
	// f(x) = x^3 - x - 2, root near 1.5213797068045676
	const auto result = BisectionSearch([](const double x) { return x * x * x - x - 2.0; }, Interval<double>{1.0, 2.0});

	REQUIRE(result.IsConverged);
	CHECK(result.Position == Approx(1.5213797068045676).margin(1e-12));
	CHECK(result.FunctionValue == Approx(0.0).margin(1e-9));
}


//----------------------------------------------------------------------------------------------------------------------
// Trivial brackets where the root is sampled before iterating
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("BisectionSearch returns immediately when left endpoint is a root", "[Math][Search]")
{
	// f(x) = x^2 - 1 on [1, 2]: f(1) == 0 exactly, so no iterations should run.
	const auto result = BisectionSearch([](const double x) { return x * x - 1.0; }, Interval<double>{1.0, 2.0});

	REQUIRE(result.IsConverged);
	CHECK(result.IterationCount == 0);
	CHECK(result.Position == 1.0);
	CHECK(result.FunctionValue == 0.0);
}

TEST_CASE("BisectionSearch returns immediately when right endpoint is a root", "[Math][Search]")
{
	// f(x) = x^2 - 1 on [-2, 1]: f(1) == 0 exactly.
	const auto result = BisectionSearch([](const double x) { return x * x - 1.0; }, Interval<double>{-2.0, 1.0});

	REQUIRE(result.IsConverged);
	CHECK(result.IterationCount == 0);
	CHECK(result.Position == 1.0);
	CHECK(result.FunctionValue == 0.0);
}


//----------------------------------------------------------------------------------------------------------------------
// Bracket precondition
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("BisectionSearch throws when the interval does not bracket a root", "[Math][Search]")
{
	// f(x) = x^2 + 1 is strictly positive on [-1, 1].
	REQUIRE_THROWS_AS((BisectionSearch([](const double x) { return x * x + 1.0; }, Interval<double>{-1.0, 1.0})),
	                  RuntimeException);
}

TEST_CASE("BisectionSearch throws when both endpoints share a negative sign", "[Math][Search]")
{
	// f(x) = -x^2 - 1 is strictly negative on [-1, 1].
	REQUIRE_THROWS_AS((BisectionSearch([](const double x) { return -x * x - 1.0; }, Interval<double>{-1.0, 1.0})),
	                  RuntimeException);
}


//----------------------------------------------------------------------------------------------------------------------
// Maximum iterations and the not-converged signal
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("BisectionSearch reports IsConverged=false when MaximumIterations is exhausted", "[Math][Search]")
{
	// Bisection on [-1, 1] needs ~log2(2 / 1e-12) ≈ 41 iterations to reach 1e-12 domain width.
	// Capping iterations well below that must yield a non-converged result.
	SearchOptions<double, double> options;
	options.MaximumIterations = 5;
	options.AbsoluteDomainTolerance = 1e-12;
	options.AbsoluteCodomainTolerance = 1e-12;

	const auto result = BisectionSearch([](const double x) { return x + 1. / 3.; }, Interval<double>{-1.0, 1.0}, options);

	REQUIRE_FALSE(result.IsConverged);
	CHECK(result.IterationCount == options.MaximumIterations);
	// The returned point must still lie inside the original bracket.
	CHECK(result.Position >= -1.0);
	CHECK(result.Position <= 1.0);
	// With 5 bisections of width 2 the interval shrinks to ~2/32, so |f| should be below that.
	CHECK(std::abs(result.FunctionValue) < 1.0);
}

TEST_CASE("BisectionSearch reports the iteration count of a converged solve", "[Math][Search]")
{
	// f(x) = x has its root exactly at 0; with width-2 bracket and 1e-12 absolute tolerance we
	// expect roughly log2(2 / 1e-12) ≈ 41 iterations, capped to MaximumIterations.
	SearchOptions<double, double> options;
	options.MaximumIterations = 100;
	options.AbsoluteDomainTolerance = 1e-12;
	options.AbsoluteCodomainTolerance = 1e-12;

	const auto result = BisectionSearch([](const double x) { return x; }, Interval<double>{-1.0, 1.0}, options);

	REQUIRE(result.IsConverged);
	CHECK(result.IterationCount <= options.MaximumIterations);
	CHECK(result.Position == Approx(0.0).margin(1e-12));
}


//----------------------------------------------------------------------------------------------------------------------
// Custom tolerances: looser tolerance converges in fewer iterations
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("BisectionSearch with loose domain tolerance converges in a single iteration", "[Math][Search]")
{
	SearchOptions<double, double> loose;
	loose.AbsoluteDomainTolerance = 1.0;  // width-2 bracket is already inside the absolute band once narrowed
	loose.AbsoluteCodomainTolerance = 0.0;

	const auto result = BisectionSearch([](const double x) { return x; }, Interval<double>{-1.0, 1.0}, loose);

	REQUIRE(result.IsConverged);
	CHECK(result.IterationCount == 1);
	CHECK(result.Position == Approx(0.0).margin(1.0));
}


//----------------------------------------------------------------------------------------------------------------------
// Generic over scalar type
//----------------------------------------------------------------------------------------------------------------------

TEMPLATE_TEST_CASE(
        "BisectionSearch works for float, double, long double", "[Math][Search][template]", float, double, long double)
{
	using Scalar = TestType;

	const auto result =
	        BisectionSearch([](const Scalar x) { return x * x - Scalar{2}; }, Interval<Scalar>{Scalar{1}, Scalar{2}});

	REQUIRE(result.IsConverged);
	CHECK(result.Position == Approx(Scalar{1.4142135623730951L}).margin(Scalar{1e-6}));
}


//----------------------------------------------------------------------------------------------------------------------
// Function-object forms: stateless lambda, captured lambda, std::function, function pointer
//----------------------------------------------------------------------------------------------------------------------

TEST_CASE("BisectionSearch accepts a variety of callable forms", "[Math][Search]")
{
	const double offset = 3.0;

	SECTION("Lambda with capture")
	{
		const auto result = BisectionSearch([&](const double x) { return x * x - offset; }, Interval<double>{1.0, 2.0});

		REQUIRE(result.IsConverged);
		CHECK(result.Position == Approx(std::sqrt(offset)).margin(1e-12));
	}

	SECTION("Generic lambda with capture")
	{
		const auto result = BisectionSearch<double, double>([&](const auto x) { return x * x - offset; }, Interval<double>{1.0, 2.0});

		REQUIRE(result.IsConverged);
		CHECK(result.Position == Approx(std::sqrt(offset)).margin(1e-12));
	}

	SECTION("std::function")
	{
		const std::function<double(double)> f = [](const double x) { return x * x - 2.0; };

		const auto result = BisectionSearch(f, Interval<double>{2.0, 1.0});

		REQUIRE(result.IsConverged);
		CHECK(result.Position == Approx(1.4142135623730951).margin(1e-12));
	}

	SECTION("Function pointer with explicit float scalars")
	{
		const auto result = BisectionSearch(&SquareMinusTwoF, Interval<float>{1.0F, 2.0F});

		REQUIRE(result.IsConverged);
		CHECK(result.Position == Approx(1.41421356F).margin(1e-5F));
	}
}


//----------------------------------------------------------------------------------------------------------------------
// GoldenSectionSearch
//----------------------------------------------------------------------------------------------------------------------

namespace
{
	double SquareOffsetThree(const double x) { return (x - 3.0) * (x - 3.0); }
}


TEST_CASE("GoldenSectionSearch minimizes (x-3)² on a bracketing interval", "[Math][Search]")
{
	const auto result = GoldenSectionSearch(
	        [](const double x) { return (x - 3.0) * (x - 3.0); }, Interval<double>{0.0, 5.0});

	REQUIRE(result.IsConverged);
	CHECK(result.Position == Approx(3.0).margin(1e-9));
	CHECK(result.FunctionValue == Approx(0.0).margin(1e-9));
	CHECK(result.IterationCount > 0);
}


TEST_CASE("GoldenSectionSearch finds a non-zero minimum", "[Math][Search]")
{
	// f(x) = (x-2)² + 5 has minimum value 5 at x = 2. This is the regression case for the
	// old IsCodomainConverged that tested |f| against zero — it would never have fired here
	// because the function value is bounded away from zero across the whole interval.
	const auto result = GoldenSectionSearch(
	        [](const double x) { return (x - 2.0) * (x - 2.0) + 5.0; }, Interval<double>{0.0, 5.0});

	REQUIRE(result.IsConverged);
	CHECK(result.Position == Approx(2.0).margin(1e-9));
	CHECK(result.FunctionValue == Approx(5.0).margin(1e-9));
}


TEST_CASE("GoldenSectionSearch maximizes when passed std::greater<>", "[Math][Search]")
{
	// Maximum of -(x-3)² + 10 is 10 at x = 3.
	const auto result = GoldenSectionSearch(
	        [](const double x) { return -(x - 3.0) * (x - 3.0) + 10.0; },
	        Interval<double>{0.0, 5.0},
	        std::greater<>{});

	REQUIRE(result.IsConverged);
	CHECK(result.Position == Approx(3.0).margin(1e-9));
	CHECK(result.FunctionValue == Approx(10.0).margin(1e-9));
}


TEST_CASE("GoldenSectionSearch reports IsConverged=false when MaximumIterations is exhausted", "[Math][Search]")
{
	// Width-5 bracket needs ~log_{1/0.618}(5 / 1e-12) ≈ 61 iterations to reach 1e-12 domain width.
	// Capping well below that must yield a non-converged result.
	SearchOptions<double, double> options;
	options.MaximumIterations = 5;
	options.AbsoluteDomainTolerance = 1e-12;
	options.AbsoluteCodomainTolerance = 1e-12;

	const auto result = GoldenSectionSearch(
	        [](const double x) { return (x - 3.0) * (x - 3.0); },
	        Interval<double>{0.0, 5.0},
	        options);

	REQUIRE_FALSE(result.IsConverged);
	CHECK(result.IterationCount == options.MaximumIterations);
	// Best estimate must still lie inside the original bracket.
	CHECK(result.Position >= 0.0);
	CHECK(result.Position <= 5.0);
}


TEST_CASE("GoldenSectionSearch accepts a variety of callable forms", "[Math][Search]")
{
	SECTION("Function pointer")
	{
		const auto result = GoldenSectionSearch(&SquareOffsetThree, Interval<double>{0.0, 5.0});

		REQUIRE(result.IsConverged);
		CHECK(result.Position == Approx(3.0).margin(1e-9));
	}

	SECTION("std::function")
	{
		const std::function<double(double)> f = [](const double x) { return (x - 3.0) * (x - 3.0); };

		const auto result = GoldenSectionSearch(f, Interval<double>{0.0, 5.0});

		REQUIRE(result.IsConverged);
		CHECK(result.Position == Approx(3.0).margin(1e-9));
	}

	SECTION("Lambda with capture")
	{
		const double offset = 3.0;

		const auto result = GoldenSectionSearch(
		        [&](const double x) { return (x - offset) * (x - offset); }, Interval<double>{0.0, 5.0});

		REQUIRE(result.IsConverged);
		CHECK(result.Position == Approx(offset).margin(1e-9));
	}

	SECTION("Generic lambda with capture")
	{
		const double offset = 3.0;

		const auto result = GoldenSectionSearch<double, double>(
		        [&](const auto x) { return (x - offset) * (x - offset); }, Interval<double>{0.0, 5.0});

		REQUIRE(result.IsConverged);
		CHECK(result.Position == Approx(offset).margin(1e-9));
	}
}
