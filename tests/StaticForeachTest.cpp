//
// Created by Andy on 2/21/2025.
//


#include <SecUtility/Meta/StaticForeach.hpp>

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <sstream>

using namespace SecUtility;

static constexpr std::array<int, 8> TestSourceArray = {1, 20, 300, 4000, 50000, 600000, 7000000, 80000000};


TEST_CASE("StaticForeach")
{
#if __cplusplus >= 202002L
	SECTION("empty range")
	{
		std::stringstream ss;
		StaticForeach(std::make_index_sequence<0>{}, [&]<std::size_t I>() { ss << I; });
		CHECK(ss.str().empty());

		int x = 42;
		StaticForeach(std::make_index_sequence<0>{}, [&]<std::size_t I>() { x += I; });
		CHECK(x == 42);
	}

	SECTION("non-empty range")
	{
		std::stringstream ss;
		StaticForeach(std::make_index_sequence<3>{}, [&]<std::size_t I>() { ss << I; });
		CHECK(ss.str() == "012");

		int x = 42;
		StaticForeach(std::make_index_sequence<3>{}, [&]<std::size_t I>() { x += TestSourceArray[I]; });
		CHECK(x == 42 + 321);
	}
#else
	SKIP("Test requires C++20");
#endif
}

TEST_CASE("StaticForeachInRange")
{
#if __cplusplus >= 202002L
	SECTION("empty range")
	{
		std::stringstream ss;
		StaticForeachInRange<0>([&]<std::size_t I>() { ss << I; });
		CHECK(ss.str().empty());

		int x = 42;
		StaticForeachInRange<0>([&]<std::size_t I>() { x += I; });
		CHECK(x == 42);
	}

	SECTION("non-empty range")
	{
		std::stringstream ss;
		StaticForeachInRange<3>([&]<std::size_t I>() { ss << I; });
		CHECK(ss.str() == "012");

		int x = 42;
		StaticForeachInRange<3>([&]<std::size_t I>() { x += TestSourceArray[I]; });
		CHECK(x == 42 + 321);
	}

	SECTION("positive lower bound, empty range")
	{
		std::stringstream ss;
		StaticForeachInRange<3, 3>([&]<std::size_t I>() { ss << I; });
		CHECK(ss.str().empty());

		int x = 42;
		StaticForeachInRange<3, 3>([&]<std::size_t I>() { x += I; });
		CHECK(x == 42);
	}

	SECTION("positive lower bound, non-empty range")
	{
		std::stringstream ss;
		StaticForeachInRange<3, 5>([&]<std::size_t I>() { ss << I; });
		CHECK(ss.str() == "34");

		int x = 42;
		StaticForeachInRange<3, 5>([&]<std::size_t I>() { x += TestSourceArray[I]; });
		CHECK(x == 42 + 54000);
	}

	SECTION("negative lower bound, empty range")
	{
		std::stringstream ss;
		StaticForeachInRange<-3, -3>([&]<int I>() { ss << I; });
		CHECK(ss.str().empty());

		int x = 42;
		StaticForeachInRange<-3, -3>([&]<int I>() { x += I; });
		CHECK(x == 42);
	}

	SECTION("negative lower bound, non-empty range")
	{
		std::stringstream ss;
		StaticForeachInRange<-3, 5>([&]<int I>() { ss << I; });
		CHECK(ss.str() == "-3-2-101234");
	}

	SECTION("custom step size, empty range")
	{
		std::stringstream ss;

		StaticForeachInRange<3, 3, 6>([&]<int I>() { ss << I; });
		CHECK(ss.str().empty());

		StaticForeachInRange<-3, -3, 6>([&]<int I>() { ss << I; });
		CHECK(ss.str().empty());

		StaticForeachInRange<3, 3, -6>([&]<int I>() { ss << I; });
		CHECK(ss.str().empty());

		StaticForeachInRange<-3, -3, -6>([&]<int I>() { ss << I; });
		CHECK(ss.str().empty());
	}

	SECTION("custom step size, non-empty range")
	{
		{
			std::stringstream ss;
			StaticForeachInRange<1, 2, 6>([&]<int I>() { ss << I; });
			CHECK(ss.str() == "1");
		}
		{
			std::stringstream ss;

			StaticForeachInRange<5, 2, -6>([&]<int I>() { ss << I; });
			CHECK(ss.str() == "5");
		}
		//------------------------------------
		{
			std::stringstream ss;
			StaticForeachInRange<-3, 5, 2>([&]<int I>() { ss << I << ' '; });
			CHECK(ss.str() == "-3 -1 1 3 ");
		}
		{
			std::stringstream ss;
			StaticForeachInRange<-3, 4, 2>([&]<int I>() { ss << I << ' '; });
			CHECK(ss.str() == "-3 -1 1 3 ");
		}
		//------------------------------------
		{
			std::stringstream ss;
			StaticForeachInRange<5, -2, -2>([&]<int I>() { ss << I << ' '; });
			CHECK(ss.str() == "5 3 1 -1 ");
		}
		{
			std::stringstream ss;
			StaticForeachInRange<3, -3, -2>([&]<int I>() { ss << I << ' '; });
			CHECK(ss.str() == "3 1 -1 ");
		}
	}
#else
	SKIP("Test requires C++20");
#endif
}

//-------------------------------------------------------------------------------------------

TEST_CASE("StaticForeachInRangeWithRuntimeIndex")
{
#if __cplusplus >= 202002L
	SECTION("positive lower bound, non-empty range")
	{
		std::stringstream ss;
		StaticForeachInRangeWithRuntimeIndex<3, 5>([&](const int i) { ss << i; });
		CHECK(ss.str() == "34");

		int x = 42;
		StaticForeachInRangeWithRuntimeIndex<3, 5>([&](const int i) { x += TestSourceArray[i]; });
		CHECK(x == 42 + 54000);
	}

	SECTION("negative lower bound, empty range")
	{
		std::stringstream ss;
		StaticForeachInRangeWithRuntimeIndex<-3, -3>([&](const int i) { ss << i; });
		CHECK(ss.str().empty());

		int x = 42;
		StaticForeachInRangeWithRuntimeIndex<-3, -3>([&](const int i) { x += i; });
		CHECK(x == 42);
	}

	SECTION("negative lower bound, non-empty range")
	{
		std::stringstream ss;
		StaticForeachInRangeWithRuntimeIndex<-3, 5>([&](const int i) { ss << i; });
		CHECK(ss.str() == "-3-2-101234");
	}
#else
SKIP("Test requires C++20");
#endif
}

// TEST_CASE("InhomogeneousStaticForeachWithRuntimeIndex w/ empty single segment")
// {
// 	std::stringstream ss;
// 	InhomogeneousStaticForeachWithRuntimeIndex<-3, -3>([&](const int i) { ss << "[0]" << i; });
// 	CHECK(ss.str().empty());
// }

// TEST_CASE("InhomogeneousStaticForeachWithRuntimeIndex w/ non-empty single segment")
// {
// 	std::stringstream ss;
// 	InhomogeneousStaticForeachWithRuntimeIndex<-3, -1>([&](const int i) { ss << "[0]" << i; });
// 	CHECK(ss.str() == "[0]-3[0]-2");
// }

// TEST_CASE("InhomogeneousStaticForeachWithRuntimeIndex w/ two non-empty segments")
// {
// 	std::stringstream ss;
// 	InhomogeneousStaticForeachWithRuntimeIndex<-3, -1, 2>([&](const int i) { ss << "[0]" << i; },
// 	                                                      [&](const int i) { ss << "[1]" << i; });
// 	CHECK(ss.str() == "[0]-3[0]-2[1]-1[1]0[1]1");
// }

// TEST_CASE("InhomogeneousStaticForeachWithRuntimeIndex w/ one empty segment, one non-empty segment")
// {
// 	std::stringstream ss;
// 	InhomogeneousStaticForeachWithRuntimeIndex<-3, -4, 2>([&](const int i) { ss << "[0]" << i; },
// 	                                                      [&](const int i) { ss << "[1]" << i; });
// 	CHECK(ss.str() == "[1]-4[1]-3[1]-2[1]-1[1]0[1]1");
// }

// TEST_CASE("InhomogeneousStaticForeachWithRuntimeIndex w/ one non-empty segment, one empty segment")
// {
// 	std::stringstream ss;
// 	InhomogeneousStaticForeachWithRuntimeIndex<-3, -1, -2>([&](const int i) { ss << "[0]" << i; },
// 	                                                       [&](const int i) { ss << "[1]" << i; });
// 	CHECK(ss.str() == "[0]-3[0]-2");
// }

// //----------------------------

// TEST_CASE("StaticForeachInRangeWithTaggedRuntimeIndex w/ empty range")
// {
// 	std::stringstream ss;
// 	StaticForeachInRangeWithTaggedRuntimeIndex<[](const int x) { return x == 0 ? 'A' : (x == 1 ? 'B' : 'C'); }, 0>(
// 	        [&]<char Tag>(const int index) { ss << Tag << index; });
// 	CHECK(ss.str().empty());
// }

// TEST_CASE("StaticForeachInRangeWithTaggedRuntimeIndex w/ non-empty range")
// {
// 	std::stringstream ss;
// 	StaticForeachInRangeWithTaggedRuntimeIndex<[](const int x) { return x == 0 ? 'A' : (x == 1 ? 'B' : 'C'); }, 5>(
// 	        [&]<char Tag>(const int index) { ss << Tag << index; });
// 	CHECK(ss.str() == "A0B1C2C3C4");
// }
