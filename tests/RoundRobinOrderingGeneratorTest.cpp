//
// Created by Andy on 10/21/2025.
//

#include <iostream>
#include <set>
#include <utility>

std::ostream& operator<<(std::ostream& os, std::pair<int, int> p)
{
	return os << '{' << p.first << ", " << p.second << '}';
}

#include <SecUtility/Collection/RoundRobinOrdering.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#include <range/v3/view/iota.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

TEST_CASE("Even")
{
	auto pairs = SecUtility::RoundRobinOrderingGenerator<int>::GenerateInitialPairs(6);
	CHECK(pairs.size() == 3);
	CHECK(pairs == std::vector<std::pair<int, int>>{{0, 3}, {1, 4}, {2, 5}});
	SecUtility::RoundRobinOrderingGenerator<int>::UpdatePairs(pairs);
	CHECK(pairs == std::vector<std::pair<int, int>>{{0, 4}, {3, 5}, {1, 2}});
	SecUtility::RoundRobinOrderingGenerator<int>::UpdatePairs(pairs);
	CHECK(pairs == std::vector<std::pair<int, int>>{{0, 5}, {4, 2}, {3, 1}});
	SecUtility::RoundRobinOrderingGenerator<int>::UpdatePairs(pairs);
	CHECK(pairs == std::vector<std::pair<int, int>>{{0, 2}, {5, 1}, {4, 3}});
	SecUtility::RoundRobinOrderingGenerator<int>::UpdatePairs(pairs);
	CHECK(pairs == std::vector<std::pair<int, int>>{{0, 1}, {2, 3}, {5, 4}});
	SecUtility::RoundRobinOrderingGenerator<int>::UpdatePairs(pairs);
	CHECK(pairs == std::vector<std::pair<int, int>>{{0, 3}, {1, 4}, {2, 5}});
}

TEST_CASE("Even2")
{
	auto pairs = SecUtility::RoundRobinOrdering{std::array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, -1};

	for (int i = 1; i < 32; i++)
	{
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 9});
		CHECK(pairs[1] == std::pair{1, 8});
		CHECK(pairs[2] == std::pair{2, 7});
		CHECK(pairs[3] == std::pair{3, 6});
		CHECK(pairs[4] == std::pair{4, 5});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 8});
		CHECK(pairs[1] == std::pair{9, 7});
		CHECK(pairs[2] == std::pair{1, 6});
		CHECK(pairs[3] == std::pair{2, 5});
		CHECK(pairs[4] == std::pair{3, 4});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 7});
		CHECK(pairs[1] == std::pair{8, 6});
		CHECK(pairs[2] == std::pair{9, 5});
		CHECK(pairs[3] == std::pair{1, 4});
		CHECK(pairs[4] == std::pair{2, 3});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 6});
		CHECK(pairs[1] == std::pair{7, 5});
		CHECK(pairs[2] == std::pair{8, 4});
		CHECK(pairs[3] == std::pair{9, 3});
		CHECK(pairs[4] == std::pair{1, 2});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 5});
		CHECK(pairs[1] == std::pair{6, 4});
		CHECK(pairs[2] == std::pair{7, 3});
		CHECK(pairs[3] == std::pair{8, 2});
		CHECK(pairs[4] == std::pair{9, 1});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 4});
		CHECK(pairs[1] == std::pair{5, 3});
		CHECK(pairs[2] == std::pair{6, 2});
		CHECK(pairs[3] == std::pair{7, 1});
		CHECK(pairs[4] == std::pair{8, 9});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 3});
		CHECK(pairs[1] == std::pair{4, 2});
		CHECK(pairs[2] == std::pair{5, 1});
		CHECK(pairs[3] == std::pair{6, 9});
		CHECK(pairs[4] == std::pair{7, 8});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 2});
		CHECK(pairs[1] == std::pair{3, 1});
		CHECK(pairs[2] == std::pair{4, 9});
		CHECK(pairs[3] == std::pair{5, 8});
		CHECK(pairs[4] == std::pair{6, 7});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 1});
		CHECK(pairs[1] == std::pair{2, 9});
		CHECK(pairs[2] == std::pair{3, 8});
		CHECK(pairs[3] == std::pair{4, 7});
		CHECK(pairs[4] == std::pair{5, 6});

		//--------------------------------------------------------------------------------------------------------------

		pairs.NextCycle();
	}
}

TEST_CASE("Even3")
{
	auto pairs = SecUtility::RoundRobinOrdering{ranges::views::iota(0, 10), -1};

	for (int i = 1; i < 32; i++)
	{
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 9});
		CHECK(pairs[1] == std::pair{1, 8});
		CHECK(pairs[2] == std::pair{2, 7});
		CHECK(pairs[3] == std::pair{3, 6});
		CHECK(pairs[4] == std::pair{4, 5});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 8});
		CHECK(pairs[1] == std::pair{9, 7});
		CHECK(pairs[2] == std::pair{1, 6});
		CHECK(pairs[3] == std::pair{2, 5});
		CHECK(pairs[4] == std::pair{3, 4});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 7});
		CHECK(pairs[1] == std::pair{8, 6});
		CHECK(pairs[2] == std::pair{9, 5});
		CHECK(pairs[3] == std::pair{1, 4});
		CHECK(pairs[4] == std::pair{2, 3});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 6});
		CHECK(pairs[1] == std::pair{7, 5});
		CHECK(pairs[2] == std::pair{8, 4});
		CHECK(pairs[3] == std::pair{9, 3});
		CHECK(pairs[4] == std::pair{1, 2});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 5});
		CHECK(pairs[1] == std::pair{6, 4});
		CHECK(pairs[2] == std::pair{7, 3});
		CHECK(pairs[3] == std::pair{8, 2});
		CHECK(pairs[4] == std::pair{9, 1});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 4});
		CHECK(pairs[1] == std::pair{5, 3});
		CHECK(pairs[2] == std::pair{6, 2});
		CHECK(pairs[3] == std::pair{7, 1});
		CHECK(pairs[4] == std::pair{8, 9});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 3});
		CHECK(pairs[1] == std::pair{4, 2});
		CHECK(pairs[2] == std::pair{5, 1});
		CHECK(pairs[3] == std::pair{6, 9});
		CHECK(pairs[4] == std::pair{7, 8});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 2});
		CHECK(pairs[1] == std::pair{3, 1});
		CHECK(pairs[2] == std::pair{4, 9});
		CHECK(pairs[3] == std::pair{5, 8});
		CHECK(pairs[4] == std::pair{6, 7});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 1});
		CHECK(pairs[1] == std::pair{2, 9});
		CHECK(pairs[2] == std::pair{3, 8});
		CHECK(pairs[3] == std::pair{4, 7});
		CHECK(pairs[4] == std::pair{5, 6});

		//--------------------------------------------------------------------------------------------------------------

		pairs.NextCycle();
	}
}

TEST_CASE("Odd")
{
	auto pairs = SecUtility::RoundRobinOrderingGenerator<int>::GenerateInitialPairs(5, 0, 9);
	CHECK(pairs.size() == 3);
	CHECK(pairs == std::vector<std::pair<int, int>>{{0, 3}, {1, 4}, {2, 9}});
	SecUtility::RoundRobinOrderingGenerator<int>::UpdatePairs(pairs);
	CHECK(pairs == std::vector<std::pair<int, int>>{{0, 4}, {3, 9}, {1, 2}});
	SecUtility::RoundRobinOrderingGenerator<int>::UpdatePairs(pairs);
	CHECK(pairs == std::vector<std::pair<int, int>>{{0, 9}, {4, 2}, {3, 1}});
	SecUtility::RoundRobinOrderingGenerator<int>::UpdatePairs(pairs);
	CHECK(pairs == std::vector<std::pair<int, int>>{{0, 2}, {9, 1}, {4, 3}});
	SecUtility::RoundRobinOrderingGenerator<int>::UpdatePairs(pairs);
	CHECK(pairs == std::vector<std::pair<int, int>>{{0, 1}, {2, 3}, {9, 4}});
	SecUtility::RoundRobinOrderingGenerator<int>::UpdatePairs(pairs);
	CHECK(pairs == std::vector<std::pair<int, int>>{{0, 3}, {1, 4}, {2, 9}});
}


TEST_CASE("Odd2")
{
	auto pairs = SecUtility::RoundRobinOrdering{std::array{0, 1, 2, 3, 4, 5, 6, 7, 8}, 9};

	for (int i = 1; i < 32; i++)
	{
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 9});
		CHECK(pairs[1] == std::pair{1, 8});
		CHECK(pairs[2] == std::pair{2, 7});
		CHECK(pairs[3] == std::pair{3, 6});
		CHECK(pairs[4] == std::pair{4, 5});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 8});
		CHECK(pairs[1] == std::pair{9, 7});
		CHECK(pairs[2] == std::pair{1, 6});
		CHECK(pairs[3] == std::pair{2, 5});
		CHECK(pairs[4] == std::pair{3, 4});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 7});
		CHECK(pairs[1] == std::pair{8, 6});
		CHECK(pairs[2] == std::pair{9, 5});
		CHECK(pairs[3] == std::pair{1, 4});
		CHECK(pairs[4] == std::pair{2, 3});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 6});
		CHECK(pairs[1] == std::pair{7, 5});
		CHECK(pairs[2] == std::pair{8, 4});
		CHECK(pairs[3] == std::pair{9, 3});
		CHECK(pairs[4] == std::pair{1, 2});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 5});
		CHECK(pairs[1] == std::pair{6, 4});
		CHECK(pairs[2] == std::pair{7, 3});
		CHECK(pairs[3] == std::pair{8, 2});
		CHECK(pairs[4] == std::pair{9, 1});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 4});
		CHECK(pairs[1] == std::pair{5, 3});
		CHECK(pairs[2] == std::pair{6, 2});
		CHECK(pairs[3] == std::pair{7, 1});
		CHECK(pairs[4] == std::pair{8, 9});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 3});
		CHECK(pairs[1] == std::pair{4, 2});
		CHECK(pairs[2] == std::pair{5, 1});
		CHECK(pairs[3] == std::pair{6, 9});
		CHECK(pairs[4] == std::pair{7, 8});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 2});
		CHECK(pairs[1] == std::pair{3, 1});
		CHECK(pairs[2] == std::pair{4, 9});
		CHECK(pairs[3] == std::pair{5, 8});
		CHECK(pairs[4] == std::pair{6, 7});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 1});
		CHECK(pairs[1] == std::pair{2, 9});
		CHECK(pairs[2] == std::pair{3, 8});
		CHECK(pairs[3] == std::pair{4, 7});
		CHECK(pairs[4] == std::pair{5, 6});

		//--------------------------------------------------------------------------------------------------------------

		pairs.NextCycle();
	}
}


TEST_CASE("Odd3")
{
	auto pairs = SecUtility::RoundRobinOrdering{std::array{0, 1, 2, 3, 4, 5, 6, 7, 8}, -1};

	for (int i = 1; i < 32; i++)
	{
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, -1});
		CHECK(pairs[1] == std::pair{1, 8});
		CHECK(pairs[2] == std::pair{2, 7});
		CHECK(pairs[3] == std::pair{3, 6});
		CHECK(pairs[4] == std::pair{4, 5});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 8});
		CHECK(pairs[1] == std::pair{-1, 7});
		CHECK(pairs[2] == std::pair{1, 6});
		CHECK(pairs[3] == std::pair{2, 5});
		CHECK(pairs[4] == std::pair{3, 4});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 7});
		CHECK(pairs[1] == std::pair{8, 6});
		CHECK(pairs[2] == std::pair{-1, 5});
		CHECK(pairs[3] == std::pair{1, 4});
		CHECK(pairs[4] == std::pair{2, 3});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 6});
		CHECK(pairs[1] == std::pair{7, 5});
		CHECK(pairs[2] == std::pair{8, 4});
		CHECK(pairs[3] == std::pair{-1, 3});
		CHECK(pairs[4] == std::pair{1, 2});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 5});
		CHECK(pairs[1] == std::pair{6, 4});
		CHECK(pairs[2] == std::pair{7, 3});
		CHECK(pairs[3] == std::pair{8, 2});
		CHECK(pairs[4] == std::pair{-1, 1});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 4});
		CHECK(pairs[1] == std::pair{5, 3});
		CHECK(pairs[2] == std::pair{6, 2});
		CHECK(pairs[3] == std::pair{7, 1});
		CHECK(pairs[4] == std::pair{8, -1});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 3});
		CHECK(pairs[1] == std::pair{4, 2});
		CHECK(pairs[2] == std::pair{5, 1});
		CHECK(pairs[3] == std::pair{6, -1});
		CHECK(pairs[4] == std::pair{7, 8});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 2});
		CHECK(pairs[1] == std::pair{3, 1});
		CHECK(pairs[2] == std::pair{4, -1});
		CHECK(pairs[3] == std::pair{5, 8});
		CHECK(pairs[4] == std::pair{6, 7});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 1});
		CHECK(pairs[1] == std::pair{2, -1});
		CHECK(pairs[2] == std::pair{3, 8});
		CHECK(pairs[3] == std::pair{4, 7});
		CHECK(pairs[4] == std::pair{5, 6});

		//--------------------------------------------------------------------------------------------------------------

		pairs.NextCycle();
	}
}


TEST_CASE("Odd4")
{
	auto pairs = SecUtility::RoundRobinOrdering{ranges::views::iota(0, 9), -1};

	for (int i = 1; i < 32; i++)
	{
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, -1});
		CHECK(pairs[1] == std::pair{1, 8});
		CHECK(pairs[2] == std::pair{2, 7});
		CHECK(pairs[3] == std::pair{3, 6});
		CHECK(pairs[4] == std::pair{4, 5});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 8});
		CHECK(pairs[1] == std::pair{-1, 7});
		CHECK(pairs[2] == std::pair{1, 6});
		CHECK(pairs[3] == std::pair{2, 5});
		CHECK(pairs[4] == std::pair{3, 4});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 7});
		CHECK(pairs[1] == std::pair{8, 6});
		CHECK(pairs[2] == std::pair{-1, 5});
		CHECK(pairs[3] == std::pair{1, 4});
		CHECK(pairs[4] == std::pair{2, 3});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 6});
		CHECK(pairs[1] == std::pair{7, 5});
		CHECK(pairs[2] == std::pair{8, 4});
		CHECK(pairs[3] == std::pair{-1, 3});
		CHECK(pairs[4] == std::pair{1, 2});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 5});
		CHECK(pairs[1] == std::pair{6, 4});
		CHECK(pairs[2] == std::pair{7, 3});
		CHECK(pairs[3] == std::pair{8, 2});
		CHECK(pairs[4] == std::pair{-1, 1});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 4});
		CHECK(pairs[1] == std::pair{5, 3});
		CHECK(pairs[2] == std::pair{6, 2});
		CHECK(pairs[3] == std::pair{7, 1});
		CHECK(pairs[4] == std::pair{8, -1});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 3});
		CHECK(pairs[1] == std::pair{4, 2});
		CHECK(pairs[2] == std::pair{5, 1});
		CHECK(pairs[3] == std::pair{6, -1});
		CHECK(pairs[4] == std::pair{7, 8});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 2});
		CHECK(pairs[1] == std::pair{3, 1});
		CHECK(pairs[2] == std::pair{4, -1});
		CHECK(pairs[3] == std::pair{5, 8});
		CHECK(pairs[4] == std::pair{6, 7});

		pairs.NextCycle();
		CHECK(pairs.PairsPerCycle() == 5);
		CHECK(pairs[0] == std::pair{0, 1});
		CHECK(pairs[1] == std::pair{2, -1});
		CHECK(pairs[2] == std::pair{3, 8});
		CHECK(pairs[3] == std::pair{4, 7});
		CHECK(pairs[4] == std::pair{5, 6});

		//--------------------------------------------------------------------------------------------------------------

		pairs.NextCycle();
	}
}

TEST_CASE("BiPartiteRoundRobinOrdering basic functionality")
{
	std::vector<int> a{0, 1, 2};
	std::vector<int> b{10, 11, 12};
	SecUtility::BiPartiteRoundRobinOrdering rr(a, b);

	SECTION("Sizes are correct")
	{
		REQUIRE(rr.PairsPerCycle() == 3);
		REQUIRE(rr.Period() == 3);
	}

	SECTION("Round 0 yields correct pairs")
	{
		auto p0 = rr[0];
		REQUIRE(p0 == std::pair{0, 10});
		auto p1 = rr[1];
		REQUIRE(p1 == std::pair{1, 11});
		auto p2 = rr[2];
		REQUIRE(p2 == std::pair{2, 12});
	}

	SECTION("Rotation works correctly")
	{
		rr.NextCycle();  // move to round 1
		REQUIRE(rr[0] == std::pair{0, 11});
		REQUIRE(rr[1] == std::pair{1, 12});
		REQUIRE(rr[2] == std::pair{2, 10});
	}

	SECTION("Multiple full rotations return to start")
	{
		for (int i = 0; i < 3; i++)
		{
			rr.NextCycle();
		}
		REQUIRE(rr[0] == std::pair{0, 10});
	}
}

TEST_CASE("BiPartiteRoundRobinOrdering asymmetric sizes")
{
	std::vector<int> a{0, 1, 2};
	std::vector<int> b{10, 11, 12, 13};
	SecUtility::BiPartiteRoundRobinOrdering rr(a, b);

	SECTION("Pairs per round is min(a,b)")
	{
		REQUIRE(rr.PairsPerCycle() == 3);
	}

	SECTION("Period is max(a,b)")
	{
		REQUIRE(rr.Period() == 4);
	}

	SECTION("All pairs eventually appear")
	{
		std::set<std::pair<int, int>> seen;
		for (int round = 0; round < rr.Period(); ++round)
		{
			for (int i = 0; i < rr.PairsPerCycle(); ++i)
			{
				seen.insert(rr[i]);
			}
			rr.NextCycle();
		}

		// All 12 pairs should appear exactly once
		std::set<std::pair<int, int>> expected{{0, 10},
		                                       {0, 11},
		                                       {0, 12},
		                                       {0, 13},
		                                       {1, 10},
		                                       {1, 11},
		                                       {1, 12},
		                                       {1, 13},
		                                       {2, 10},
		                                       {2, 11},
		                                       {2, 12},
		                                       {2, 13}};
		REQUIRE(seen == expected);
	}
}

TEST_CASE("BiPartiteRoundRobinOrdering multiple passes with rotation")
{
	std::vector<int> a{0, 1};
	std::vector<int> b{100, 101, 102};
	SecUtility::BiPartiteRoundRobinOrdering rr(a, b);

	SECTION("Rotation offset accumulates correctly")
	{
		rr.NextCycle();  // round 1
		REQUIRE(rr[0] == std::pair{0, 101});
		REQUIRE(rr[1] == std::pair{1, 102});

		rr.NextCycle();  // round 2
		REQUIRE(rr[0] == std::pair{0, 102});
		REQUIRE(rr[1] == std::pair{1, 100});

		rr.NextCycle();  // round 3
		REQUIRE(rr[0] == std::pair{0, 100});
		REQUIRE(rr[1] == std::pair{1, 101});
	}
}

TEST_CASE("BiPartiteRoundRobinOrdering first range larger than second")
{
	std::vector<int> a{0, 1, 2, 3};
	std::vector<int> b{10, 11};
	SecUtility::BiPartiteRoundRobinOrdering rr(a, b);

	SECTION("Pairs per round is min(a,b)")
	{
		REQUIRE(rr.PairsPerCycle() == 2);
	}

	SECTION("Period is max(a,b)")
	{
		REQUIRE(rr.Period() == 4);
	}

	SECTION("Round 0 yields correct pairs")
	{
		REQUIRE(rr[0] == std::pair{0, 10});
		REQUIRE(rr[1] == std::pair{1, 11});
	}

	SECTION("Rotation walks the larger range while the smaller stays fixed")
	{
		rr.NextCycle();  // round 1
		REQUIRE(rr[0] == std::pair{1, 10});
		REQUIRE(rr[1] == std::pair{2, 11});

		rr.NextCycle();  // round 2
		REQUIRE(rr[0] == std::pair{2, 10});
		REQUIRE(rr[1] == std::pair{3, 11});

		rr.NextCycle();  // round 3
		REQUIRE(rr[0] == std::pair{3, 10});
		REQUIRE(rr[1] == std::pair{0, 11});
	}

	SECTION("All pairs eventually appear")
	{
		std::set<std::pair<int, int>> seen;
		for (int round = 0; round < rr.Period(); ++round)
		{
			for (int i = 0; i < rr.PairsPerCycle(); ++i)
			{
				seen.insert(rr[i]);
			}
			rr.NextCycle();
		}

		std::set<std::pair<int, int>> expected{{0, 10},
		                                       {0, 11},
		                                       {1, 10},
		                                       {1, 11},
		                                       {2, 10},
		                                       {2, 11},
		                                       {3, 10},
		                                       {3, 11}};
		REQUIRE(seen == expected);
	}
}