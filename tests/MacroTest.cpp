//
// Created by Andy on 2/12/2026.
//

#include <SecUtility/Macro/ConstevalIf.hpp>
#include <catch2/catch_test_macros.hpp>


TEST_CASE("if consteval macro should work")
{
#if defined(SEC_IF_CONSTEVAL)
	constexpr auto f = []
	{
		SEC_IF_CONSTEVAL
		{
			return 4.2;
		}
		else
		{
			return 6.9;
		}
	};

	constexpr auto g = []
	{
		SEC_IF_NOT_CONSTEVAL
		{
			return 6.9;
		}
		else
		{
			return 4.2;
		}
	};

	STATIC_CHECK(f() == 4.2);
	CHECK(f() == 6.9);
	STATIC_CHECK(g() == 4.2);
	CHECK(g() == 6.9);
#else
	SKIP("Test requires C++20 or compiler magic");
#endif
}
