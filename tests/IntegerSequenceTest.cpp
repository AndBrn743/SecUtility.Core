//
// Created by Andy on 4/24/2026.
//

#include <SecUtility/Meta/IntegerSequence.hpp>
#include <catch2/catch_test_macros.hpp>
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
