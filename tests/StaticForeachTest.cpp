//
// Created by Andy on 2/21/2025.
//


// Exercise deprecated entry points too. GCC also diagnoses their internal calls
// at the template definition, so suppress the warning before including the header.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

#include <SecUtility/Meta/StaticForeach.hpp>

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace SecUtility;

namespace
{
	struct TemplateRecorder
	{
		std::vector<int>& visited;

		template <auto I>
		void operator()() &
		{
			visited.push_back(I);
		}
	};

	struct RuntimeRecorder
	{
		std::vector<int>& visited;
		void operator()(int i) &
		{
			visited.push_back(i);
		}
	};

	struct TemplateSum
	{
		int& sum;

		template <auto I>
		constexpr void operator()() noexcept
		{
			sum += I;
		}
	};

	struct TemplateValue
	{
		template <auto I>
		int operator()() const
		{
			return I;
		}
	};

	struct RuntimeValue
	{
		int operator()(int i) const
		{
			return i;
		}
	};

	struct IndexDependentReturn
	{
		template <int I>
		auto operator()() const -> std::conditional_t<I == 0, void, int>;
	};

	struct IndexDependentNoexcept
	{
		template <int I>
		void operator()() const noexcept(I == 0)
		{
		}
	};

	struct ThrowingTemplateRecorder
	{
		std::vector<int>& visited;

		template <int I>
		void operator()()
		{
			visited.push_back(I);
			if constexpr (I == 1)
			{
				throw std::runtime_error("stop iteration");
			}
		}
	};

	template <typename Sequence, typename Operation, typename = void>
	struct CanStaticForeach : std::false_type
	{
	};

	template <typename Sequence, typename Operation>
	struct CanStaticForeach<Sequence,
	                        Operation,
	                        std::void_t<decltype(StaticForeach(std::declval<Sequence>(), std::declval<Operation>()))>>
	    : std::true_type
	{
	};

	template <typename Operation, typename, auto... Bounds>
	struct CanStaticForeachInRange : std::false_type
	{
	};

	template <typename Operation, auto... Bounds>
	struct CanStaticForeachInRange<Operation,
	                               std::void_t<decltype(StaticForeachInRange<Bounds...>(std::declval<Operation>()))>,
	                               Bounds...> : std::true_type
	{
	};

	template <typename Operation, auto... Bounds>
	constexpr bool CanIterateRange = CanStaticForeachInRange<Operation, void, Bounds...>::value;

	template <auto... Bounds>
	void CheckRange(const std::vector<int>& expected)
	{
		std::vector<int> visited;
		StaticForeachInRange<Bounds...>(TemplateRecorder{visited});
		CHECK(visited == expected);
		visited.clear();
		StaticForeachInRange<Bounds...>(RuntimeRecorder{visited});
		CHECK(visited == expected);
	}

	constexpr int TemplateSumAtCompileTime()
	{
		int sum = 0;
		StaticForeach(std::integer_sequence<int, -2, 4, 4>{}, TemplateSum{sum});
		StaticForeachInRange<4>(TemplateSum{sum});
		return sum;
	}

	constexpr int RuntimeSumAtCompileTime()
	{
		int sum = 0;
		auto add = [&](int i) constexpr noexcept { sum += i; };
		StaticForeach(std::integer_sequence<int, -2, 4, 4>{}, add);
		StaticForeachInRange<4>(add);
		return sum;
	}
}

TEST_CASE("StaticForeach preserves sequence order and duplicates")
{
	std::vector<int> visited;
	using Sequence = std::integer_sequence<int, 4, -2, 4, 0>;
	SECTION("template index")
	{
		StaticForeach(Sequence{}, TemplateRecorder{visited});
	}
	SECTION("runtime index")
	{
		StaticForeach(Sequence{}, RuntimeRecorder{visited});
	}
	CHECK(visited == std::vector<int>{4, -2, 4, 0});
}

TEST_CASE("StaticForeach empty sequences do not require a callable")
{
	STATIC_CHECK(noexcept(StaticForeach(std::integer_sequence<int>{}, 42)));
	STATIC_CHECK(noexcept(StaticForeachInRange<0>(42)));
	StaticForeach(std::integer_sequence<int>{}, 42);
	StaticForeachInRange<0>(42);

	int calls = 0;
	auto record = [&](int) { ++calls; };
	StaticForeach(std::integer_sequence<int>{}, record);
	StaticForeachInRange<0>(record);
	CHECK(calls == 0);
}

TEST_CASE("StaticForeachInRange visits zero through upper bound exclusively")
{
	std::vector<int> visited;
	std::vector<int> expected = {0};
	SECTION("template index, singleton")
	{
		StaticForeachInRange<1>(TemplateRecorder{visited});
	}
	SECTION("runtime index, singleton")
	{
		StaticForeachInRange<1>(RuntimeRecorder{visited});
	}
	SECTION("template index, multiple indices")
	{
		StaticForeachInRange<4>(TemplateRecorder{visited});
		expected = {0, 1, 2, 3};
	}
	SECTION("runtime index, multiple indices")
	{
		StaticForeachInRange<4>(RuntimeRecorder{visited});
		expected = {0, 1, 2, 3};
	}
	CHECK(visited == expected);
}

TEST_CASE("StaticForeach preserves runtime index types")
{
	int calls = 0;
	StaticForeach(std::integer_sequence<short, -1, 2>{},
	              [&](auto i)
	              {
		              STATIC_CHECK(std::is_same_v<decltype(i), short>);
		              ++calls;
	              });
	StaticForeachInRange<3u>(
	        [&](auto i)
	        {
		        STATIC_CHECK(std::is_same_v<decltype(i), unsigned int>);
		        ++calls;
	        });
	CHECK(calls == 5);
}

TEST_CASE("StaticForeach requires void results at every index")
{
	using Sequence = std::integer_sequence<int, 0, 1>;
	STATIC_CHECK(CanStaticForeach<Sequence, TemplateSum>::value);
	STATIC_CHECK(CanStaticForeach<Sequence, RuntimeRecorder>::value);
	STATIC_CHECK_FALSE(CanStaticForeach<Sequence, TemplateValue>::value);
	STATIC_CHECK_FALSE(CanStaticForeach<Sequence, RuntimeValue>::value);
	STATIC_CHECK_FALSE(CanStaticForeach<Sequence, IndexDependentReturn>::value);
	STATIC_CHECK(CanStaticForeach<std::integer_sequence<int, 0>, IndexDependentReturn>::value);
	STATIC_CHECK_FALSE(CanStaticForeach<Sequence, int>::value);
}

TEST_CASE("StaticForeach supports constexpr evaluation and propagates noexcept")
{
	STATIC_CHECK(TemplateSumAtCompileTime() == 12);
	STATIC_CHECK(RuntimeSumAtCompileTime() == 12);
	STATIC_CHECK(noexcept(StaticForeach(std::make_index_sequence<3>{}, std::declval<TemplateSum>())));
	STATIC_CHECK(noexcept(StaticForeachInRange<3>(std::declval<TemplateSum>())));
	STATIC_CHECK(noexcept(StaticForeach(std::integer_sequence<int, 0>{}, IndexDependentNoexcept{})));
	STATIC_CHECK_FALSE(noexcept(StaticForeach(std::integer_sequence<int, 0, 1>{}, IndexDependentNoexcept{})));
	STATIC_CHECK_FALSE(noexcept(StaticForeachInRange<2>(IndexDependentNoexcept{})));

	auto nonthrowing = [](int) noexcept {};
	auto throwing = [](int) {};
	STATIC_CHECK(noexcept(StaticForeach(std::make_index_sequence<3>{}, nonthrowing)));
	STATIC_CHECK(noexcept(StaticForeachInRange<3>(nonthrowing)));
	STATIC_CHECK_FALSE(noexcept(StaticForeach(std::make_index_sequence<3>{}, throwing)));
	STATIC_CHECK_FALSE(noexcept(StaticForeachInRange<3>(throwing)));
}

TEST_CASE("StaticForeach propagates exceptions and stops at the throwing index")
{
	std::vector<int> visited;
	SECTION("template index")
	{
		CHECK_THROWS_AS(StaticForeachInRange<3>(ThrowingTemplateRecorder{visited}), std::runtime_error);
	}
	SECTION("runtime index")
	{
		CHECK_THROWS_AS(StaticForeachInRange<3>(
		                        [&](int i)
		                        {
			                        visited.push_back(i);
			                        if (i == 1)
			                        {
				                        throw std::runtime_error("stop iteration");
			                        }
		                        }),
		                std::runtime_error);
	}
	CHECK(visited == std::vector<int>{0, 1});
}

TEST_CASE("StaticForeach invokes move-only lambdas without copying or moving them")
{
	int sum = 0;
	auto add = [state = std::make_unique<int>(0), &sum](int i) mutable
	{
		*state += i;
		sum = *state;
	};
	StaticForeachInRange<3>(add);
	CHECK(sum == 3);
	StaticForeachInRange<3>(std::move(add));
	CHECK(sum == 6);
	StaticForeachInRange<2, 4>(std::move(add));
	CHECK(sum == 11);
	StaticForeachInRange<5, 0, -2>(std::move(add));
	CHECK(sum == 20);
	add(1);
	CHECK(sum == 21);
}

TEST_CASE("StaticForeachInRange bounded and stepped forms work in C++17")
{
	CheckRange<3, 5>({3, 4});
	CheckRange<-3, 2>({-3, -2, -1, 0, 1});
	CheckRange<0, 1>({0});
	CheckRange<-3, -3>({});
	CheckRange<-3, 5, 2>({-3, -1, 1, 3});
	CheckRange<-3, 4, 2>({-3, -1, 1, 3});
	CheckRange<5, -2, -2>({5, 3, 1, -1});
	CheckRange<3, -3, -2>({3, 1, -1});
	CheckRange<1, 2, 6>({1});
	CheckRange<5, 2, -6>({5});
	CheckRange<3, 3, 6>({});
	CheckRange<-3, -3, -6>({});
}

TEST_CASE("StaticForeachInRange keeps step arithmetic independent of bound types")
{
	CheckRange<5u, 1u, -2>({5, 3});
	CheckRange<5u, 0u, -2>({5, 3, 1});
	CheckRange<1u, 5u, 2>({1, 3});
	CheckRange<5u, 1, -2>({5, 3});
	CheckRange<-3, 4, 2u>({-3, -1, 1, 3});
	CheckRange<1u, 4u>({1, 2, 3});
	CheckRange<static_cast<signed char>(-128), static_cast<signed char>(127), 127>({-128, -1, 126});
	int calls = 0;
	StaticForeachInRange<5u, 1u, -2>(
	        [&](auto i)
	        {
		        STATIC_CHECK(std::is_same_v<decltype(i), unsigned>);
		        ++calls;
	        });
	StaticForeachInRange<static_cast<short>(1), static_cast<short>(4), 2LL>(
	        [&](auto i)
	        {
		        STATIC_CHECK(std::is_same_v<decltype(i), short>);
		        ++calls;
	        });
	CHECK(calls == 4);
}

TEST_CASE("StaticForeachInRange handles extreme bounds without signed overflow")
{
	constexpr auto Min = std::numeric_limits<std::intmax_t>::min();
	constexpr auto Max = std::numeric_limits<std::intmax_t>::max();
	std::vector<std::intmax_t> visited;
	StaticForeachInRange<Min, Max, Max>([&](auto i) { visited.push_back(i); });
	CHECK(visited == std::vector<std::intmax_t>{Min, -1, Max - 1});
	visited.clear();
	StaticForeachInRange<Max, Min, Min>([&](auto i) { visited.push_back(i); });
	CHECK(visited == std::vector<std::intmax_t>{Max, -1});
}

TEST_CASE("StaticForeachInRange consistently constrains ranges and callbacks")
{
	STATIC_CHECK(CanIterateRange<TemplateSum, 0, 3>);
	STATIC_CHECK(CanIterateRange<RuntimeRecorder, 3u, 0u, -1>);
	STATIC_CHECK_FALSE(CanIterateRange<TemplateValue, 3>);
	STATIC_CHECK_FALSE(CanIterateRange<TemplateValue, 0, 3>);
	STATIC_CHECK_FALSE(CanIterateRange<RuntimeValue, 0, 3>);
	STATIC_CHECK_FALSE(CanIterateRange<RuntimeValue, 3, 0, -1>);
	STATIC_CHECK_FALSE(CanIterateRange<IndexDependentReturn, 0, 2>);
	STATIC_CHECK_FALSE(CanIterateRange<TemplateSum, -1>);
	STATIC_CHECK_FALSE(CanIterateRange<TemplateSum, 3, 1>);
	STATIC_CHECK_FALSE(CanIterateRange<TemplateSum, 0, 3, 0>);
	STATIC_CHECK_FALSE(CanIterateRange<TemplateSum, 0, 3, -1>);
	// The common bound type is unsigned, which cannot represent the negative bound.
	STATIC_CHECK_FALSE(CanIterateRange<TemplateSum, -3, 4u>);
	STATIC_CHECK_FALSE(CanIterateRange<TemplateSum, -3, 4u, 2>);
	STATIC_CHECK(CanIterateRange<int, -3, -3>);
	STATIC_CHECK(CanIterateRange<int, 3u, 3u, -1>);
	StaticForeachInRange<-3, -3>(42);
	StaticForeachInRange<3u, 3u, -1>(42);
}

TEST_CASE("StaticForeachInRange bounded forms preserve constexpr and noexcept")
{
	constexpr auto templateSum = []() constexpr
	{
		int sum = 0;
		StaticForeachInRange<-2, 2>(TemplateSum{sum});
		StaticForeachInRange<5, -2, -2>(TemplateSum{sum});
		return sum;
	};
	constexpr auto runtimeSum = []() constexpr
	{
		int sum = 0;
		auto add = [&](int i) constexpr noexcept { sum += i; };
		StaticForeachInRange<-2, 2>(add);
		StaticForeachInRange<5, -2, -2>(add);
		return sum;
	};
	STATIC_CHECK(templateSum() == 6);
	STATIC_CHECK(runtimeSum() == 6);
	STATIC_CHECK(noexcept(StaticForeachInRange<-2, 2>(std::declval<TemplateSum>())));
	STATIC_CHECK(noexcept(StaticForeachInRange<5, -2, -2>(std::declval<TemplateSum>())));
	STATIC_CHECK(noexcept(StaticForeachInRange<-3, -3>(42)));
	STATIC_CHECK(noexcept(StaticForeachInRange<3, 3, -1>(42)));
	STATIC_CHECK_FALSE(noexcept(StaticForeachInRange<0, 2>(IndexDependentNoexcept{})));
	STATIC_CHECK_FALSE(noexcept(StaticForeachInRange<1, -1, -1>(IndexDependentNoexcept{})));
	auto nonthrowing = [](int) noexcept {};
	auto throwing = [](int) {};
	STATIC_CHECK(noexcept(StaticForeachInRange<-2, 2>(nonthrowing)));
	STATIC_CHECK(noexcept(StaticForeachInRange<5, -2, -2>(nonthrowing)));
	STATIC_CHECK_FALSE(noexcept(StaticForeachInRange<-2, 2>(throwing)));
	STATIC_CHECK_FALSE(noexcept(StaticForeachInRange<5, -2, -2>(throwing)));
}

TEST_CASE("StaticForeachInRange bounded forms propagate exceptions")
{
	std::vector<int> visited;
	SECTION("template index")
	{
		CHECK_THROWS_AS((StaticForeachInRange<0, 3>(ThrowingTemplateRecorder{visited})), std::runtime_error);
	}
	SECTION("runtime index")
	{
		auto record = [&](int i)
		{
			visited.push_back(i);
			if (i == 1)
			{
				throw std::runtime_error("stop iteration");
			}
		};
		CHECK_THROWS_AS((StaticForeachInRange<0, 3, 1>(record)), std::runtime_error);
	}
	CHECK(visited == std::vector<int>{0, 1});
}


#if defined(__cpp_generic_lambdas) && __cpp_generic_lambdas >= 201707L
static constexpr std::array<int, 8> TestSourceArray = {1, 20, 300, 4000, 50000, 600000, 7000000, 80000000};

TEST_CASE("StaticForeach with template lambdas")
{
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
}

TEST_CASE("StaticForeachInRange with template lambdas")
{
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
}

#endif

#if defined(__cpp_generic_lambdas) && __cpp_generic_lambdas >= 201707L
TEST_CASE("StaticForeachInRangeWithRuntimeIndex")
{
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
}
#endif

TEST_CASE("Legacy runtime-index entry points")
{
	std::vector<int> visited;
	auto record = [&](int i) { visited.push_back(i); };
	StaticForeachWithRuntimeIndex(std::integer_sequence<int>{}, record);
	StaticForeachInRangeWithRuntimeIndex<0>(record);
	CHECK(visited.empty());
	StaticForeachWithRuntimeIndex(std::integer_sequence<int, 4, -2, 4>{}, record);
	CHECK(visited == std::vector<int>{4, -2, 4});
	visited.clear();
	StaticForeachInRangeWithRuntimeIndex<3>(record);
	CHECK(visited == std::vector<int>{0, 1, 2});
	visited.clear();
#if defined(__cpp_generic_lambdas) && __cpp_generic_lambdas >= 201707L
	StaticForeachInRangeWithRuntimeIndex<-2, 2>(record);
	CHECK(visited == std::vector<int>{-2, -1, 0, 1});
	visited.clear();
#endif
	// Legacy callbacks may still return values that are discarded.
	// StaticForeachInRangeWithRuntimeIndex<1, 3>(
	//         [&](int i)
	//         {
	// 	        visited.push_back(i);
	// 	        return i;
	//         });
	// CHECK(visited == std::vector<int>{1, 2});
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
