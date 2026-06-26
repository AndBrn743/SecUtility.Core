//
// Created by Andy on 5/30/2026.
//

#include <SecUtility/Meta/NttpArray.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <utility>
#include <SecUtility/Diagnostic/Unreachable.hpp>


#if __cplusplus >= 202002L
struct MyInt
{
	[[maybe_unused]] int Value;
};


template <auto...>
class MyClass
{
	/* NO CODE */
};

template <typename, typename, auto...>
class MyClassToo
{
	/* NO CODE */
};

template <typename... Ts>
struct BindTypes
{
	template <auto... Vs>
	using Apply = MyClassToo<Ts..., Vs...>;
};

template <auto... Vs>
using BindTypesToo = MyClassToo<char, long double, Vs...>;
#endif

using namespace SecUtility;

TEST_CASE("NttpArray")
{
#if __cplusplus >= 202002L
	{
		constexpr std::array NttpArray{MyInt{42}, MyInt{69}};
		using Result = ApplyNttpArrayTo<MyClass, NttpArray>;
		STATIC_CHECK(std::is_same_v<Result, MyClass<MyInt{42}, MyInt{69}>>);
	}
	{
		constexpr std::array NttpArray{MyInt{42}, MyInt{69}};
		using Result = ApplyNttpArrayTo<BindTypes<int, double>::Apply, NttpArray>;
		STATIC_CHECK(std::is_same_v<Result, MyClassToo<int, double, MyInt{42}, MyInt{69}>>);
	}
	{
		constexpr std::array NttpArray{MyInt{42}, MyInt{69}};
		using Result = ApplyNttpArrayTo<BindTypesToo, NttpArray>;
		STATIC_CHECK(std::is_same_v<Result, MyClassToo<char, long double, MyInt{42}, MyInt{69}>>);
	}
	{
		constexpr std::array NttpArray{MyInt{42}, MyInt{69}};
		using Result = ApplyNttpArrayToGenerator<decltype([]<auto... Vs>() -> MyClassToo<char, long double, Vs...> { Unreachable(); }), NttpArray>;
		STATIC_CHECK(std::is_same_v<Result, MyClassToo<char, long double, MyInt{42}, MyInt{69}>>);
	}
#else
	SKIP("Feature require C++20");
#endif
}
