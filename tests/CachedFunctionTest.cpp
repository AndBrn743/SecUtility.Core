//
// Created by Claude on 2/9/2026.
//

#include <SecUtility/Misc/CachedFunction.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <string>
#include <vector>

using namespace SecUtility;


// Helper function to track call counts
int globalCallCount = 0;

int ExpensiveFunction(int x)
{
	++globalCallCount;
	return x * x;
}

int AddFunction(int a, int b)
{
	++globalCallCount;
	return a + b;
}

std::string ConcatFunction(const std::string& a, const std::string& b)
{
	++globalCallCount;
	return a + b;
}


TEST_CASE("CachedFunction - Basic caching with single argument")
{
	globalCallCount = 0;
	CachedFunction<int(int)> cached(ExpensiveFunction);

	SECTION("First call executes the function")
	{
		int result = cached(5);
		CHECK(result == 25);
		CHECK(globalCallCount == 1);
	}

	SECTION("Subsequent calls with same argument use cache")
	{
		cached(5);
		CHECK(globalCallCount == 1);

		cached(5);  // Should use cache
		CHECK(globalCallCount == 1);  // No additional call

		cached(5);  // Should use cache
		CHECK(globalCallCount == 1);
	}

	SECTION("Different arguments execute the function")
	{
		cached(5);
		CHECK(globalCallCount == 1);

		cached(10);  // Different argument
		CHECK(globalCallCount == 2);

		cached(5);   // Should use cache
		CHECK(globalCallCount == 2);

		cached(10);  // Should use cache
		CHECK(globalCallCount == 2);
	}

	SECTION("Cache stores multiple results")
	{
		cached(1);
		cached(2);
		cached(3);
		CHECK(globalCallCount == 3);

		CHECK(cached(1) == 1);
		CHECK(cached(2) == 4);
		CHECK(cached(3) == 9);
		CHECK(globalCallCount == 3);  // All cached
	}
}


TEST_CASE("CachedFunction - Multiple arguments")
{
	globalCallCount = 0;
	CachedFunction<int(int, int)> cached(AddFunction);

	SECTION("Caching with multiple arguments")
	{
		int result = cached(3, 4);
		CHECK(result == 7);
		CHECK(globalCallCount == 1);

		// Same arguments - should use cache
		result = cached(3, 4);
		CHECK(result == 7);
		CHECK(globalCallCount == 1);
	}

	SECTION("Different argument combinations")
	{
		cached(1, 2);
		cached(2, 1);
		cached(1, 2);  // Should use cache
		cached(2, 2);

		CHECK(globalCallCount == 3);  // Only 3 unique calls
	}
}


TEST_CASE("CachedFunction - String arguments")
{
	globalCallCount = 0;
	CachedFunction<std::string(const std::string&, const std::string&)> cached(ConcatFunction);

	SECTION("String caching works correctly")
	{
		std::string result = cached("Hello", "World");
		CHECK(result == "HelloWorld");
		CHECK(globalCallCount == 1);

		// Same arguments - should use cache
		result = cached("Hello", "World");
		CHECK(result == "HelloWorld");
		CHECK(globalCallCount == 1);

		// Different arguments - should call function
		result = cached("Goodbye", "World");
		CHECK(result == "GoodbyeWorld");
		CHECK(globalCallCount == 2);
	}

	SECTION("Different string combinations")
	{
		cached("A", "B");
		cached("B", "A");
		cached("A", "B");  // Should use cache

		CHECK(globalCallCount == 2);
	}
}


TEST_CASE("CachedFunction - Size method")
{
	globalCallCount = 0;
	CachedFunction<int(int)> cached(ExpensiveFunction);

	SECTION("Initial cache is empty")
	{
		CHECK(cached.Size() == 0);
	}

	SECTION("Size increases with unique calls")
	{
		cached(1);
		CHECK(cached.Size() == 1);

		cached(2);
		CHECK(cached.Size() == 2);

		cached(3);
		CHECK(cached.Size() == 3);
	}

	SECTION("Size does not increase for cached calls")
	{
		cached(1);
		cached(2);
		CHECK(cached.Size() == 2);

		cached(1);  // Cached
		cached(2);  // Cached
		CHECK(cached.Size() == 2);

		cached(3);  // New
		CHECK(cached.Size() == 3);
	}
}


TEST_CASE("CachedFunction - Clear method")
{
	globalCallCount = 0;
	CachedFunction<int(int)> cached(ExpensiveFunction);

	SECTION("Clear empties the cache")
	{
		cached(1);
		cached(2);
		cached(3);
		CHECK(cached.Size() == 3);
		CHECK(globalCallCount == 3);

		cached.Clear();
		CHECK(cached.Size() == 0);

		// After clearing, should call function again
		cached(1);
		CHECK(globalCallCount == 4);
		CHECK(cached.Size() == 1);
	}

	SECTION("Clear on empty cache is safe")
	{
		CHECK(cached.Size() == 0);
		cached.Clear();
		CHECK(cached.Size() == 0);
	}
}


TEST_CASE("CachedFunction - Const operator() for cache lookups")
{
	globalCallCount = 0;
	CachedFunction<int(int)> cached(ExpensiveFunction);

	SECTION("Const operator throws on cache miss")
	{
		// Cache is empty, const call should throw
		const auto& constCached = cached;
		CHECK_THROWS_AS(constCached(5), std::out_of_range);
	}

	SECTION("Const operator works on cache hit")
	{
		// Populate cache
		cached(5);
		cached(10);
		CHECK(globalCallCount == 2);

		// Const lookup should work
		const auto& constCached = cached;
		CHECK(constCached(5) == 25);
		CHECK(constCached(10) == 100);
		CHECK(globalCallCount == 2);  // No additional calls
	}

	SECTION("Const operator does not modify cache")
	{
		cached(5);
		CHECK(cached.Size() == 1);

		const auto& constCached = cached;
		constCached(5);
		CHECK(cached.Size() == 1);  // Size unchanged
	}
}


TEST_CASE("CachedFunction - Lambda functions")
{
	globalCallCount = 0;

	SECTION("Lambda with capture")
	{
		int multiplier = 3;
		auto lambda = [&multiplier](int x) {
			++globalCallCount;
			return x * multiplier;
		};

		CachedFunction<int(int)> cached(lambda);

		CHECK(cached(5) == 15);
		CHECK(globalCallCount == 1);

		multiplier = 5;
		// Cached result is still 15, not 25
		CHECK(cached(5) == 15);
		CHECK(globalCallCount == 1);
	}

	SECTION("Stateless lambda")
	{
		auto lambda = [](int x) {
			++globalCallCount;
			return x * 2;
		};

		CachedFunction<int(int)> cached(lambda);

		CHECK(cached(10) == 20);
		CHECK(cached(10) == 20);
		CHECK(globalCallCount == 1);
	}
}


TEST_CASE("CachedFunction - Function object")
{
	globalCallCount = 0;

	struct Functor
	{
		int& counter;

		int operator()(int x) const
		{
			++counter;
			return x + 10;
		}
	};

	Functor functor{globalCallCount};
	CachedFunction<int(int)> cached(functor);

	CHECK(cached(5) == 15);
	CHECK(globalCallCount == 1);

	CHECK(cached(5) == 15);
	CHECK(globalCallCount == 1);
}


TEST_CASE("CachedFunction - Complex types")
{
	globalCallCount = 0;

	SECTION("Vector return type")
	{
		auto makeVector = [](int n) {
			++globalCallCount;
			return std::vector<int>(n, 42);
		};

		CachedFunction<std::vector<int>(int)> cached(makeVector);

		auto v1 = cached(5);
		CHECK(v1.size() == 5);
		CHECK(globalCallCount == 1);

		auto v2 = cached(5);
		CHECK(v2.size() == 5);
		CHECK(globalCallCount == 1);
	}
}


TEST_CASE("CachedFunction - No arguments function")
{
	globalCallCount = 0;

	auto noArgFunc = []() {
		++globalCallCount;
		return 42;
	};

	CachedFunction<int()> cached(noArgFunc);

	CHECK(cached() == 42);
	CHECK(globalCallCount == 1);

	CHECK(cached() == 42);
	CHECK(globalCallCount == 1);

	CHECK(cached.Size() == 1);
}


TEST_CASE("CachedFunction - Exception safety")
{
	globalCallCount = 0;

	auto throwingFunc = [](int x) -> int {
		++globalCallCount;
		if (x < 0)
		{
			throw std::runtime_error("Negative value");
		}
		return x * 2;
	};

	CachedFunction<int(int)> cached(throwingFunc);

	SECTION("Exception does not cache result")
	{
		CHECK_THROWS(cached(-5));
		CHECK(globalCallCount == 1);

		// Call again - should execute function again
		CHECK_THROWS(cached(-5));
		CHECK(globalCallCount == 2);
	}

	SECTION("Successful call after exception")
	{
		CHECK_THROWS(cached(-5));
		CHECK(globalCallCount == 1);

		CHECK(cached(5) == 10);
		CHECK(globalCallCount == 2);

		// Cached
		CHECK(cached(5) == 10);
		CHECK(globalCallCount == 2);
	}
}


TEST_CASE("CachedFunction - Move semantics")
{
	globalCallCount = 0;

	auto func = [](int x) {
		++globalCallCount;
		return std::string(x, 'A');
	};

	CachedFunction<std::string(int)> cached(func);

	std::string result = cached(5);
	CHECK(result == "AAAAA");
	CHECK(globalCallCount == 1);

	CHECK(cached.Size() == 1);
}
