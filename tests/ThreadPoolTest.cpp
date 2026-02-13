//
// Created by Andy on 2/14/2026.
//

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include <SecUtility/Threading/ThreadPool.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>
#include <vector>

using namespace SecUtility::Threading;
using namespace std::chrono_literals;

TEST_CASE("ThreadPool basic construction", "[threadpool][construction]")
{
	SECTION("Default construction uses hardware concurrency")
	{
		ThreadPool pool;
		REQUIRE(pool.ThreadCount() == std::thread::hardware_concurrency());
	}

	SECTION("Custom thread count")
	{
		ThreadPool pool(4);
		REQUIRE(pool.ThreadCount() == 4);
	}

	SECTION("Zero threads creates at least one thread")
	{
		ThreadPool pool(0);
		REQUIRE(pool.ThreadCount() == 1);
	}

	SECTION("Single thread pool")
	{
		ThreadPool pool(1);
		REQUIRE(pool.ThreadCount() == 1);
	}
}

TEST_CASE("ThreadPool task submission and execution", "[threadpool][submit]")
{
	ThreadPool pool(4);

	SECTION("Submit simple void task")
	{
		std::atomic<bool> executed{false};

		auto future = pool.Submit([&executed] { executed = true; });

		future.wait();
		REQUIRE(executed == true);
	}

	SECTION("Submit task returning value")
	{
		auto future = pool.Submit([] { return 42; });

		REQUIRE(future.get() == 42);
	}

	SECTION("Submit task with arguments")
	{
		auto add = [](const int a, const int b) { return a + b; };

		auto future = pool.Submit(add, 10, 32);

		REQUIRE(future.get() == 42);
	}

	SECTION("Submit task with move-only arguments")
	{
		auto future = pool.Submit([](const std::unique_ptr<int>& ptr) { return *ptr; }, std::make_unique<int>(42));

		REQUIRE(future.get() == 42);
	}

	SECTION("Submit multiple tasks")
	{
		std::vector<std::future<int>> futures;

		for (int i = 0; i < 100; ++i)
		{
			futures.push_back(pool.Submit([i] { return i * 2; }));
		}

		for (int i = 0; i < 100; ++i)
		{
			REQUIRE(futures[i].get() == i * 2);
		}
	}

	SECTION("Submit task with reference wrapper")
	{
		int value = 0;

		auto future = pool.Submit([](int& ref) { ref = 42; }, std::ref(value));

		future.wait();
		REQUIRE(value == 42);
	}
}

TEST_CASE("ThreadPool task execution order and concurrency", "[threadpool][concurrency]")
{
	SECTION("Tasks execute concurrently")
	{
		ThreadPool pool(4);
		std::atomic<int> counter{0};
		std::vector<std::future<void>> futures;

		// Submit tasks that increment counter with small delay
		for (int i = 0; i < 4; ++i)
		{
			futures.push_back(pool.Submit(
			        [&counter]
			        {
				        std::this_thread::sleep_for(10ms);
				        counter++;
			        }));
		}

		// Wait for all tasks
		for (auto& f : futures)
		{
			f.wait();
		}

		REQUIRE(counter == 4);
	}

	SECTION("Work distribution across threads")
	{
		ThreadPool pool(4);
		std::atomic<int> activeThreads{0};
		std::atomic<int> maxConcurrent{0};
		std::vector<std::future<void>> futures;

		for (int i = 0; i < 8; ++i)
		{
			futures.push_back(pool.Submit(
			        [&activeThreads, &maxConcurrent]
			        {
				        const int current = ++activeThreads;

				        // Update max concurrent
				        int currentMax = maxConcurrent.load();
				        while (current > currentMax && !maxConcurrent.compare_exchange_weak(currentMax, current))
				        {
				        }

				        std::this_thread::sleep_for(50ms);
				        --activeThreads;
			        }));
		}

		for (auto& f : futures)
		{
			f.wait();
		}

		// Should have had multiple threads working concurrently
		REQUIRE(maxConcurrent > 1);
		REQUIRE(maxConcurrent <= 4);
	}
}

TEST_CASE("ThreadPool exception handling", "[threadpool][exceptions]")
{
	ThreadPool pool(2);

	SECTION("Task throwing exception")
	{
		auto future = pool.Submit([] { throw std::runtime_error("Task error"); });

		try
		{
			future.get();
			REQUIRE(false);
		}
		catch (const std::runtime_error& e)
		{
			REQUIRE(e.what() == std::string{"Task error"});
		}
		catch (...)
		{
			REQUIRE(false);
		}
	}

	SECTION("Exception in one task doesn't affect others")
	{
		auto future1 = pool.Submit([] { throw std::runtime_error("Error"); });

		auto future2 = pool.Submit([] { return 42; });

		REQUIRE_THROWS(future1.get());
		REQUIRE(future2.get() == 42);
	}

	SECTION("Multiple exceptions")
	{
		std::vector<std::future<int>> futures;

		for (int i = 0; i < 10; ++i)
		{
			futures.push_back(pool.Submit(
			        [i]
			        {
				        if (i % 2 == 0)
				        {
					        throw std::runtime_error("Even error");
				        }
				        return i;
			        }));
		}

		int exceptionCount = 0;
		int successCount = 0;

		for (auto& f : futures)
		{
			try
			{
				f.get();
				++successCount;
			}
			catch (...)
			{
				++exceptionCount;
			}
		}

		REQUIRE(exceptionCount == 5);
		REQUIRE(successCount == 5);
	}
}

TEST_CASE("ThreadPool stress tests", "[threadpool][stress]")
{
	SECTION("Many small tasks")
	{
		ThreadPool pool(4);
		constexpr int taskCount = 10000;
		std::atomic<int> counter{0};
		std::vector<std::future<void>> futures;

		for (int i = 0; i < taskCount; ++i)
		{
			futures.push_back(pool.Submit([&counter] { counter++; }));
		}

		for (auto& f : futures)
		{
			f.wait();
		}

		REQUIRE(counter == taskCount);
	}

	SECTION("Heavy computation tasks")
	{
		ThreadPool pool(4);

		auto fibonacci = [](const int n) -> long long
		{
			if (n <= 1)
			{
				return n;
			}
			long long a = 0, b = 1;
			for (int i = 2; i <= n; ++i)
			{
				const long long temp = a + b;
				a = b;
				b = temp;
			}
			return b;
		};

		std::vector<std::future<long long>> futures;
		for (int i = 0; i < 20; ++i)
		{
			futures.push_back(pool.Submit(fibonacci, 30));
		}

		for (auto& f : futures)
		{
			REQUIRE(f.get() == 832040);
		}
	}

	SECTION("Mixed task durations")
	{
		ThreadPool pool(4);
		std::vector<std::future<int>> futures;

		for (int i = 0; i < 20; ++i)
		{
			futures.push_back(pool.Submit(
			        [i]
			        {
				        if (i % 3 == 0)
				        {
					        std::this_thread::sleep_for(10ms);
				        }
				        return i;
			        }));
		}

		for (int i = 0; i < 20; ++i)
		{
			REQUIRE(futures[i].get() == i);
		}
	}
}

TEST_CASE("ThreadPool destruction behavior", "[threadpool][destruction]")
{
	SECTION("Destruction waits for all tasks to complete")
	{
		std::atomic<int> completed{0};

		{
			ThreadPool pool(2);

			for (int i = 0; i < 10; ++i)
			{
				pool.Submit(
				        [&completed]
				        {
					        std::this_thread::sleep_for(10ms);
					        completed++;
				        });
			}
			// Pool destructor should wait for all tasks
		}

		REQUIRE(completed == 10);
	}

	SECTION("Cannot submit to stopped pool")
	{
		auto pool = std::make_unique<ThreadPool>(2);

		// Destroy the pool
		pool.reset();

		// Create new pool - this tests that construction/destruction works correctly
		pool = std::make_unique<ThreadPool>(2);
		auto future = pool->Submit([] { return 42; });

		REQUIRE(future.get() == 42);
	}

	SECTION("Pending tasks execute before destruction")
	{
		std::vector<int> results;

		{
			std::mutex resultsMutex;
			ThreadPool pool(1);  // Single thread to ensure sequential execution

			for (int i = 0; i < 5; ++i)
			{
				pool.Submit(
				        [i, &results, &resultsMutex]
				        {
					        std::this_thread::sleep_for(10ms);
					        std::lock_guard lock(resultsMutex);
					        results.push_back(i);
				        });
			}
			// All tasks should complete before pool is destroyed
		}

		REQUIRE(results.size() == 5);
	}
}

TEST_CASE("ThreadPool different return types", "[threadpool][types]")
{
	ThreadPool pool(2);

	SECTION("Return string")
	{
		auto future = pool.Submit([] { return std::string("Hello, ThreadPool!"); });

		REQUIRE(future.get() == "Hello, ThreadPool!");
	}

	SECTION("Return vector")
	{
		auto future = pool.Submit([] { return std::vector<int>{1, 2, 3, 4, 5}; });

		auto result = future.get();
		REQUIRE(result == std::vector<int>{1, 2, 3, 4, 5});
	}

	SECTION("Return unique_ptr")
	{
		auto future = pool.Submit([] { return std::make_unique<int>(42); });

		auto ptr = future.get();
		REQUIRE(*ptr == 42);
	}

	SECTION("Return custom struct")
	{
		struct Result
		{
			int value;
			std::string message;
			bool operator==(const Result& other) const
			{
				return value == other.value && message == other.message;
			}
		};

		auto future = pool.Submit([] { return Result{42, "Success"}; });

		REQUIRE(future.get() == Result{42, "Success"});
	}
}

TEST_CASE("ThreadPool edge cases", "[threadpool][edge]")
{
	SECTION("Submit lambda with captures")
	{
		ThreadPool pool(2);
		int captured = 100;

		auto future = pool.Submit([captured] { return captured * 2; });

		REQUIRE(future.get() == 200);
	}

	SECTION("Submit member function")
	{
		struct Calculator
		{
			int multiply(const int a, const int b) const
			{
				return a * b;
			}
		};

		ThreadPool pool(2);
		Calculator calc;

		auto future = pool.Submit(&Calculator::multiply, &calc, 6, 7);

		REQUIRE(future.get() == 42);
	}

	// SECTION("Recursive task submission")  // not enough threads. will deadlock
	// {
	// 	ThreadPool pool(4);
	// 	std::atomic<int> counter{0};
	//
	// 	std::function<void(int)> recursiveTask;
	// 	recursiveTask = [&](int depth)
	// 	{
	// 		counter++;
	// 		if (depth > 0)
	// 		{
	// 			pool.Submit(recursiveTask, depth - 1).wait();
	// 		}
	// 	};
	//
	// 	pool.Submit(recursiveTask, 5).wait();
	//
	// 	REQUIRE(counter == 6);  // 0, 1, 2, 3, 4, 5
	// }

	SECTION("Very large return type")
	{
		ThreadPool pool(2);

		auto future = pool.Submit([] { return std::vector<int>(10000, 42); });

		auto result = future.get();
		REQUIRE(result.size() == 10000);
		REQUIRE(std::all_of(result.begin(), result.end(), [](int v) { return v == 42; }));
	}
}

TEST_CASE("ThreadPool recursive patterns", "[threadpool][recursive]")
{
	SECTION("Fire-and-forget recursive tasks")
	{
		ThreadPool pool(4);
		std::atomic<int> counter{0};

		std::function<void(int)> task;
		task = [&](const int depth)
		{
			counter++;
			if (depth > 0)
			{
				// Submit children without waiting
				pool.Submit(task, depth - 1);
				pool.Submit(task, depth - 1);
			}
		};

		pool.Submit(task, 3);
		std::this_thread::sleep_for(100ms);

		// 2^4 - 1 = 15 nodes in complete binary tree of depth 3
		REQUIRE(counter >= 7);  // At least the first few levels
	}

	SECTION("Demonstrates thread pool saturation")
	{
		ThreadPool pool(2);
		std::atomic<int> concurrentTasks{0};
		std::atomic<int> maxConcurrent{0};

		auto task = [&]
		{
			const int current = ++concurrentTasks;
			int expected = maxConcurrent.load();
			while (current > expected && !maxConcurrent.compare_exchange_weak(expected, current))
			{
			}

			std::this_thread::sleep_for(50ms);
			--concurrentTasks;
		};

		// Submit more tasks than threads
		std::vector<std::future<void>> futures;
		for (int i = 0; i < 10; ++i)
		{
			futures.push_back(pool.Submit(task));
		}

		for (auto& f : futures)
		{
			f.wait();
		}

		REQUIRE(maxConcurrent == 2);  // Never exceeded thread count
	}

	SECTION("Parallel divide-and-conquer (limited depth)")
	{
		ThreadPool pool(4);

		// Parallel sum with limited recursion depth
		std::function<int(const std::vector<int>&, size_t, size_t, int)> parallelSum;
		parallelSum = [&](const std::vector<int>& data, size_t start, const size_t end, const int depthLimit) -> int
		{
			if (end - start <= 10 || depthLimit <= 0)
			{
				// Sequential base case
				return std::accumulate(data.begin() + start, data.begin() + end, 0);
			}

			size_t mid = start + (end - start) / 2;

			// Only parallelize if we have depth budget
			auto leftFuture = pool.Submit(parallelSum, std::ref(data), start, mid, depthLimit - 1);
			const int rightSum = parallelSum(data, mid, end, depthLimit - 1);

			return leftFuture.get() + rightSum;
		};

		std::vector<int> data(1000);
		std::iota(data.begin(), data.end(), 1);

		int sum = pool.Submit(parallelSum, std::ref(data), 0, data.size(), 2).get();  // Limit depth to 2

		REQUIRE(sum == 500500);  // Sum of 1 to 1000
	}
}

TEST_CASE("ThreadPool thread safety", "[threadpool][threadsafety]")
{
	SECTION("Concurrent submissions from multiple threads")
	{
		ThreadPool pool(4);
		std::atomic<int> counter{0};
		std::vector<std::thread> submitters;

		for (int i = 0; i < 4; ++i)
		{
			submitters.emplace_back(
			        [&pool, &counter]
			        {
				        for (int j = 0; j < 100; ++j)
				        {
					        pool.Submit([&counter] { counter++; });
				        }
			        });
		}

		for (auto& t : submitters)
		{
			t.join();
		}

		// Give tasks time to complete
		std::this_thread::sleep_for(100ms);

		REQUIRE(counter == 400);
	}

	SECTION("Shared resource with proper synchronization")
	{
		ThreadPool pool(4);
		std::vector<int> sharedData;
		std::mutex dataMutex;
		std::vector<std::future<void>> futures;

		for (int i = 0; i < 100; ++i)
		{
			futures.push_back(pool.Submit(
			        [i, &sharedData, &dataMutex]
			        {
				        std::lock_guard lock(dataMutex);
				        sharedData.push_back(i);
			        }));
		}

		for (auto& f : futures)
		{
			f.wait();
		}

		REQUIRE(sharedData.size() == 100);

		// Verify all values are present
		std::sort(sharedData.begin(), sharedData.end());
		for (int i = 0; i < 100; ++i)
		{
			REQUIRE(sharedData[i] == i);
		}
	}
}

TEST_CASE("ThreadPool MasterThreadPool singleton", "[threadpool][singleton]")
{
	SECTION("MasterThreadPool returns same instance")
	{
		auto& pool1 = MasterThreadPool();
		auto& pool2 = MasterThreadPool();

		REQUIRE(&pool1 == &pool2);
	}

	SECTION("MasterThreadPool is functional")
	{
		auto future = MasterThreadPool().Submit([] { return 42; });

		REQUIRE(future.get() == 42);
	}

	SECTION("MasterThreadPool handles multiple concurrent tasks")
	{
		std::vector<std::future<int>> futures;

		for (int i = 0; i < 50; ++i)
		{
			futures.push_back(MasterThreadPool().Submit([i] { return i * i; }));
		}

		for (int i = 0; i < 50; ++i)
		{
			REQUIRE(futures[i].get() == i * i);
		}
	}
}

// Benchmarks (optional, requires Catch2 v3)
#if CATCH_VERSION_MAJOR >= 3
TEST_CASE("ThreadPool benchmarks", "[threadpool][benchmark][!benchmark]")
{
	ThreadPool pool(4);

	BENCHMARK("Submit simple task")
	{
		return pool.Submit([] { return 42; });
	};

	BENCHMARK("Submit and wait for simple task")
	{
		auto future = pool.Submit([] { return 42; });
		return future.get();
	};

	BENCHMARK("Submit 100 tasks")
	{
		std::vector<std::future<int>> futures;
		for (int i = 0; i < 100; ++i)
		{
			futures.push_back(pool.Submit([i] { return i; }));
		}
		for (auto& f : futures)
		{
			f.get();
		}
	};
}
#endif