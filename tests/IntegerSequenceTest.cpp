//
// Created by Andy on 4/24/2026.
//

#include <SecUtility/Meta/IntegerSequence.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>
#include <type_traits>

TEST_CASE("IntegerSequence")
{
	STATIC_CHECK(std::is_same_v<SecUtility::MakeIndexSequence<0>, std::make_index_sequence<0>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeIndexSequence<1>, std::make_index_sequence<1>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeIndexSequence<2>, std::make_index_sequence<2>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeIndexSequence<3>, std::make_index_sequence<3>>);

	STATIC_CHECK(std::is_same_v<SecUtility::MakeIntegerSequence<int, 0>, std::make_integer_sequence<int, 0>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeIntegerSequence<int, 1>, std::make_integer_sequence<int, 1>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeIntegerSequence<int, 2>, std::make_integer_sequence<int, 2>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeIntegerSequence<int, 3>, std::make_integer_sequence<int, 3>>);

	STATIC_CHECK(std::is_same_v<SecUtility::MakeReversedIndexSequence<0>, std::index_sequence<>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeReversedIndexSequence<1>, std::index_sequence<0>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeReversedIndexSequence<2>, std::index_sequence<1, 0>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeReversedIndexSequence<3>, std::index_sequence<2, 1, 0>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeReversedIndexSequence<4>, std::index_sequence<3, 2, 1, 0>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeReversedIndexSequence<5>, std::index_sequence<4, 3, 2, 1, 0>>);

	STATIC_CHECK(std::is_same_v<SecUtility::MakeReversedIntegerSequence<int, 0>, std::integer_sequence<int>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeReversedIntegerSequence<int, 1>, std::integer_sequence<int, 0>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeReversedIntegerSequence<int, 2>, std::integer_sequence<int, 1, 0>>);
	STATIC_CHECK(std::is_same_v<SecUtility::MakeReversedIntegerSequence<int, 3>, std::integer_sequence<int, 2, 1, 0>>);
}

TEST_CASE("MakeIntegerSequence creates end-exclusive ranges")
{
	using namespace SecUtility;
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, -3, 2>, std::integer_sequence<int, -3, -2, -1, 0, 1>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, 2, 5>, std::integer_sequence<int, 2, 3, 4>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, -3, 5, 2>, std::integer_sequence<int, -3, -1, 1, 3>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, -3, 4, 2>, std::integer_sequence<int, -3, -1, 1, 3>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, 5, -2, -2>, std::integer_sequence<int, 5, 3, 1, -1>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, 3, -3, -2>, std::integer_sequence<int, 3, 1, -1>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, -1, -5, -1>, std::integer_sequence<int, -1, -2, -3, -4>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, 1, 2, 6>, std::integer_sequence<int, 1>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, 5, 2, -6>, std::integer_sequence<int, 5>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, 0, 1>, std::integer_sequence<int, 0>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, 0>, std::integer_sequence<int>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, -3, -3>, std::integer_sequence<int>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, 3, 3, 6>, std::integer_sequence<int>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, -3, -3, -6>, std::integer_sequence<int>>);
}

TEST_CASE("MakeIntegerSequence preserves its specified integer type")
{
	using namespace SecUtility;
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<short, 3>, std::integer_sequence<short, 0, 1, 2>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<unsigned long long, 0>, std::integer_sequence<unsigned long long>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<unsigned long long, 3>,
	                            std::integer_sequence<unsigned long long, 0, 1, 2>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<long, -1, 2>, std::integer_sequence<long, -1, 0, 1>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<char, 1, 3>, std::integer_sequence<char, 1, 2>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<unsigned, 5u, 1u, -2>, std::integer_sequence<unsigned, 5, 3>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<int, -3, 4u, 2u>, std::integer_sequence<int, -3, -1, 1, 3>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<signed char, -128, 127, 127>,
	                            std::integer_sequence<signed char, -128, -1, 126>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<unsigned char, 255, 0, -128>,
	                            std::integer_sequence<unsigned char, 255, 127>>);
	STATIC_CHECK(
	        std::is_same_v<MakeIntegerSequence<unsigned char, 1, 3, 65536u>, std::integer_sequence<unsigned char, 1>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<bool, 0>, std::integer_sequence<bool>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<bool, 1>, std::integer_sequence<bool, false>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<bool, true, false, -1>, std::integer_sequence<bool, true>>);
}

TEST_CASE("MakeIndexSequence uses size_t for all range forms")
{
	using namespace SecUtility;
	STATIC_CHECK(std::is_same_v<MakeIndexSequence<2, 5>, std::index_sequence<2, 3, 4>>);
	STATIC_CHECK(std::is_same_v<MakeIndexSequence<1, 8, 3>, std::index_sequence<1, 4, 7>>);
	STATIC_CHECK(std::is_same_v<MakeIndexSequence<1, 7, 3>, std::index_sequence<1, 4>>);
	STATIC_CHECK(std::is_same_v<MakeIndexSequence<5u, 1u, -2>, std::index_sequence<5, 3>>);
	STATIC_CHECK(std::is_same_v<MakeIndexSequence<4, 0, -1>, std::index_sequence<4, 3, 2, 1>>);
	STATIC_CHECK(std::is_same_v<MakeIndexSequence<0, 1>, std::index_sequence<0>>);
	STATIC_CHECK(std::is_same_v<MakeIndexSequence<1, 2, 6>, std::index_sequence<1>>);
	STATIC_CHECK(std::is_same_v<MakeIndexSequence<5, 2, -6>, std::index_sequence<5>>);
	STATIC_CHECK(std::is_same_v<MakeIndexSequence<3, 3>, std::index_sequence<>>);
	STATIC_CHECK(std::is_same_v<MakeIndexSequence<3, 3, 6>, std::index_sequence<>>);
	STATIC_CHECK(std::is_same_v<MakeIndexSequence<3, 3, -6>, std::index_sequence<>>);
}

TEST_CASE("Integer ranges avoid overflow at integer limits")
{
	using namespace SecUtility;
	constexpr auto Min = std::numeric_limits<std::intmax_t>::min();
	constexpr auto Max = std::numeric_limits<std::intmax_t>::max();
	constexpr auto UnsignedMax = std::numeric_limits<std::uintmax_t>::max();
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<std::intmax_t, Min, Max, Max>,
	                            std::integer_sequence<std::intmax_t, Min, -1, Max - 1>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<std::intmax_t, Max, Min, Min>,
	                            std::integer_sequence<std::intmax_t, Max, -1>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<std::intmax_t, Min, Max, UnsignedMax>,
	                            std::integer_sequence<std::intmax_t, Min>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<std::intmax_t, Min, Min + 2>,
	                            std::integer_sequence<std::intmax_t, Min, Min + 1>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<std::intmax_t, Max - 2, Max>,
	                            std::integer_sequence<std::intmax_t, Max - 2, Max - 1>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<std::uintmax_t, 0, UnsignedMax, UnsignedMax - 1>,
	                            std::integer_sequence<std::uintmax_t, 0, UnsignedMax - 1>>);
	STATIC_CHECK(std::is_same_v<MakeIntegerSequence<std::uintmax_t, UnsignedMax, 0, Min>,
	                            std::integer_sequence<std::uintmax_t, UnsignedMax, static_cast<std::uintmax_t>(Max)>>);
	constexpr auto IndexMax = std::numeric_limits<std::size_t>::max();
	STATIC_CHECK(
	        std::is_same_v<MakeIndexSequence<IndexMax - 2, IndexMax>, std::index_sequence<IndexMax - 2, IndexMax - 1>>);
	STATIC_CHECK(
	        std::is_same_v<MakeIndexSequence<IndexMax, IndexMax - 2, -1>, std::index_sequence<IndexMax, IndexMax - 1>>);
}
