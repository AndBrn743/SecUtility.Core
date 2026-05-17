//
// Created by Andy on 5/17/2026.
//

#include <SecUtility/Meta/TypeTrait.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace SecUtility;

struct Foo
{
	void operator()(int) const;
	void operator()(double, double) const;
};

TEST_CASE("FunctionTraits")
{
	auto a = [] {};
	auto b = [](int) {};
	auto c = [](int, double) -> float { return {}; };

	STATIC_CHECK(FunctionTraits<decltype(a)>::Arity == 0);
	STATIC_CHECK(FunctionTraits<decltype(b)>::Arity == 1);
	STATIC_CHECK(FunctionTraits<decltype(c)>::Arity == 2);

	STATIC_CHECK(std::is_same_v<FunctionTraits<decltype(a)>::ReturnType, void>);
	STATIC_CHECK(std::is_same_v<FunctionTraits<decltype(b)>::ReturnType, void>);
	STATIC_CHECK(std::is_same_v<FunctionTraits<decltype(c)>::ReturnType, float>);

	STATIC_CHECK(std::is_same_v<FunctionTraits<decltype(a)>::ArgTypeTuple, TypeTuple<>>);
	STATIC_CHECK(std::is_same_v<FunctionTraits<decltype(b)>::ArgTypeTuple, TypeTuple<int>>);
	STATIC_CHECK(std::is_same_v<FunctionTraits<decltype(c)>::ArgTypeTuple, TypeTuple<int, double>>);

	// STATIC_CHECK(FunctionTraits<Foo>::Arity == 0);  // won't compile by design
}
