//
// Created by Andy on 5/16/2026.
//

#include <SecUtility/Meta/Identity.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("SecUtility::Identity")
{
	SecUtility::Identity f;

	CHECK(f(42) == 42);
	CHECK(f(3.5) == 3.5);

	STATIC_CHECK(std::is_empty_v<SecUtility::Identity>);
	STATIC_CHECK(std::is_trivially_copyable_v<SecUtility::Identity>);
	STATIC_CHECK(std::is_default_constructible_v<SecUtility::Identity>);

	STATIC_CHECK(std::is_same_v<decltype(SecUtility::Identity{}(std::declval<int&>())), int&>);
	STATIC_CHECK(std::is_same_v<decltype(SecUtility::Identity{}(std::declval<int&&>())), int&&>);

	struct MoveOnly
	{
		MoveOnly() = default;
		MoveOnly(const MoveOnly&) = delete;
		MoveOnly(MoveOnly&&) = default;
	};

	MoveOnly m;

	STATIC_CHECK(std::is_same_v<decltype(f(std::move(m))), MoveOnly&&>);
}
