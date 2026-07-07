// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Misc/NullMutex.hpp>
#include <SecUtility/Misc/Prefetch.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <vector>

using namespace SecUtility;

namespace
{
	// Verifies arbitrary user types work as pointees (const void* conversion).
	struct Widget
	{
		int value;
		double data[4];
	};
}

TEST_CASE("Prefetch accepts stack pointers")
{
	int x = 42;
	Prefetch(&x);

	Widget w{};
	Prefetch(&w);

	int arr[8]{};
	Prefetch(arr);
	Prefetch(arr + 4);

	CHECK(x == 42);
}

TEST_CASE("Prefetch accepts heap pointers")
{
	const std::vector<int> v(1024, 7);
	Prefetch(v.data());
	Prefetch(v.data() + v.size() / 2);

	CHECK(v.front() == 7);
}

TEST_CASE("Prefetch tolerates nullptr")
{
	Prefetch(nullptr);
}

TEST_CASE("Prefetch issues many hints in a loop")
{
	std::array<int, 1024> arr{};
	for (std::size_t i = 0; i < arr.size(); ++i)
	{
		Prefetch(arr.data() + i);
		arr[i] = static_cast<int>(i);
	}

	CHECK(arr.back() == static_cast<int>(arr.size() - 1));
}

TEST_CASE("PrefetchStream accepts stack pointers")
{
	int x = 7;
	PrefetchStream(&x);
	CHECK(x == 7);
}

TEST_CASE("PrefetchStream accepts heap pointers")
{
	const std::vector<double> v(256, 3.14);
	PrefetchStream(v.data());
	PrefetchStream(v.data() + 100);

	CHECK(v.front() == 3.14);
}

TEST_CASE("PrefetchStream tolerates nullptr")
{
	PrefetchStream(nullptr);
}

TEST_CASE("Prefetch and PrefetchStream can be mixed")
{
	std::vector<int> v(512);
	for (std::size_t i = 0; i + 64 < v.size(); ++i)
	{
		Prefetch(v.data() + i);
		PrefetchStream(v.data() + i + 64);
		v[i] = static_cast<int>(i);
	}

	CHECK(v[100] == 100);
}

TEST_CASE("Prefetch functions are noexcept")
{
	int x = 0;
	STATIC_CHECK(noexcept(Prefetch(&x)));
	STATIC_CHECK(noexcept(PrefetchStream(&x)));
}

#if defined(SEC_IF_NOT_CONSTEVAL)
TEST_CASE("Prefetch is usable in constant expressions")
{
	constexpr int value = []
	{
		int v = 42;
		Prefetch(&v);
		PrefetchStream(&v);
		return v;
	}();

	STATIC_CHECK(value == 42);
}
#endif


// === NullMutex tests ===

TEST_CASE("NullMutex is an empty, trivially destructible type")
{
	STATIC_CHECK(std::is_empty_v<NullMutex>);
	STATIC_CHECK(std::is_default_constructible_v<NullMutex>);
	STATIC_CHECK(std::is_trivially_destructible_v<NullMutex>);
	STATIC_CHECK(std::is_nothrow_default_constructible_v<NullMutex>);
	STATIC_CHECK(std::is_nothrow_destructible_v<NullMutex>);
}

TEST_CASE("NullMutex satisfies Lockable requirements")
{
	NullMutex m;

	SECTION("lock and unlock are callable")
	{
		m.lock();
		m.unlock();
	}

	SECTION("try_lock always succeeds")
	{
		CHECK(m.try_lock());
		m.unlock();
	}

	SECTION("Repeated locking is permitted (no-op semantics)")
	{
		m.lock();
		m.lock();
		m.unlock();
		m.unlock();
	}
}

TEST_CASE("NullMutex satisfies SharedLockable requirements")
{
	NullMutex m;

	SECTION("lock_shared and unlock_shared are callable")
	{
		m.lock_shared();
		m.unlock_shared();
	}

	SECTION("try_lock_shared always succeeds")
	{
		CHECK(m.try_lock_shared());
		m.unlock_shared();
	}

	SECTION("Shared and exclusive operations can interleave")
	{
		m.lock();
		m.lock_shared();
		m.unlock_shared();
		m.unlock();
	}
}

TEST_CASE("NullMutex operations are noexcept")
{
	NullMutex m;
	STATIC_CHECK(noexcept(m.lock()));
	STATIC_CHECK(noexcept(m.unlock()));
	STATIC_CHECK(noexcept(m.try_lock()));
	STATIC_CHECK(noexcept(m.lock_shared()));
	STATIC_CHECK(noexcept(m.unlock_shared()));
	STATIC_CHECK(noexcept(m.try_lock_shared()));
}

TEST_CASE("NullMutex works with std synchronization primitives")
{
	NullMutex m;

	SECTION("std::lock_guard")
	{
		std::lock_guard<NullMutex> lg(m);
	}

	SECTION("std::lock_guard with adopt_lock")
	{
		m.lock();
		std::lock_guard<NullMutex> lg(m, std::adopt_lock);
	}

	SECTION("std::unique_lock supports manual lock/unlock")
	{
		std::unique_lock<NullMutex> ul(m);
		CHECK(ul.owns_lock());

		ul.unlock();
		CHECK(!ul.owns_lock());

		ul.lock();
		CHECK(ul.owns_lock());
	}

	SECTION("std::unique_lock with defer_lock")
	{
		std::unique_lock<NullMutex> ul(m, std::defer_lock);
		CHECK(!ul.owns_lock());
		ul.lock();
		CHECK(ul.owns_lock());
	}

	SECTION("std::unique_lock with try_to_lock")
	{
		std::unique_lock<NullMutex> ul(m, std::try_to_lock);
		CHECK(ul.owns_lock());
	}

	SECTION("std::shared_lock supports manual lock/unlock")
	{
		std::shared_lock<NullMutex> sl(m);
		CHECK(sl.owns_lock());

		sl.unlock();
		CHECK(!sl.owns_lock());

		sl.lock();
		CHECK(sl.owns_lock());
	}
}

TEST_CASE("NullMutex is usable in constant expressions")
{
	constexpr bool ok = []
	{
		NullMutex m;
		m.lock();
		m.unlock();
		m.lock_shared();
		m.unlock_shared();
		return m.try_lock() && m.try_lock_shared();
	}();
	STATIC_CHECK(ok);
}
