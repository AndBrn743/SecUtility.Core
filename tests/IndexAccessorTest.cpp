// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Collection/IndexAccessor.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <vector>
#include <string>


using namespace SecUtility;


TEST_CASE("MakeIndexAccessor - std::vector by value (pointer)")
{
	SECTION("Access elements using call operator")
	{
		std::vector vec{1, 2, 3, 4, 5};
		int* ptr = vec.data();

		auto accessor = MakeIndexAccessor(ptr);

		CHECK(accessor(0) == 1);
		CHECK(accessor(1) == 2);
		CHECK(accessor(2) == 3);
		CHECK(accessor(3) == 4);
		CHECK(accessor(4) == 5);
	}

	SECTION("Modify elements through accessor")
	{
		std::vector vec{1, 2, 3, 4, 5};
		int* ptr = vec.data();

		auto accessor = MakeIndexAccessor(ptr);
		accessor(2) = 42;

		CHECK(vec[2] == 42);
	}

	SECTION("Constexpr context")
	{
		constexpr auto dummy = []() constexpr
		{
			constexpr int arr[] = {10, 20, 30};
			auto accessor = MakeIndexAccessor(static_cast<const int*>(arr));

			assert(accessor(0) == 10);
			assert(accessor(1) == 20);
			assert(accessor(2) == 30);

			return accessor(0) + accessor(1) + accessor(2);
		}();

		(void)dummy;
	}
}


TEST_CASE("MakeIndexAccessor - std::vector by reference")
{
	SECTION("Access elements using call operator")
	{
		std::vector vec{1, 2, 3, 4, 5};

		auto accessor = MakeIndexAccessor(std::ref(vec));

		CHECK(accessor(0) == 1);
		CHECK(accessor(1) == 2);
		CHECK(accessor(2) == 3);
		CHECK(accessor(3) == 4);
		CHECK(accessor(4) == 5);
	}

	SECTION("Modify elements through accessor")
	{
		std::vector vec{1, 2, 3, 4, 5};

		auto accessor = MakeIndexAccessor(std::ref(vec));
		accessor(2) = 42;

		CHECK(vec[2] == 42);
	}

	SECTION("Accessor reflects changes in original vector")
	{
		std::vector vec{1, 2, 3};

		auto accessor = MakeIndexAccessor(std::ref(vec));

		CHECK(accessor(1) == 2);

		vec[1] = 99;
		CHECK(accessor(1) == 99);

		vec.push_back(4);
		CHECK(accessor(3) == 4);
	}
}


TEST_CASE("MakeIndexAccessor - std::array")
{
	SECTION("By value with const array")
	{
		constexpr std::array<int, 5> arr{1, 2, 3, 4, 5};
		const auto accessor = MakeIndexAccessor(arr);

		CHECK(accessor(0) == 1);
		CHECK(accessor(2) == 3);
		CHECK(accessor(4) == 5);
	}

	SECTION("By reference")
	{
		std::array<int, 3> arr{10, 20, 30};

		auto accessor = MakeIndexAccessor(std::ref(arr));

		CHECK(accessor(0) == 10);
		CHECK(accessor(1) == 20);
		CHECK(accessor(2) == 30);
	}
}


TEST_CASE("MakeIndexAccessor - C-style array")
{
	SECTION("Non-const array by pointer")
	{
		int arr[] = {100, 200, 300};

		auto accessor = MakeIndexAccessor(arr);

		CHECK(accessor(0) == 100);
		CHECK(accessor(1) == 200);
		CHECK(accessor(2) == 300);

		accessor(1) = 250;
		CHECK(arr[1] == 250);
	}

	SECTION("Const array by pointer")
	{
		constexpr int arr[] = {100, 200, 300};
		auto accessor = MakeIndexAccessor(arr);

		CHECK(accessor(0) == 100);
		CHECK(accessor(1) == 200);
		CHECK(accessor(2) == 300);
	}
}


TEST_CASE("MakeIndexAccessor - Different index types")
{
	SECTION("Index with int")
	{
		std::vector vec{1, 2, 3};
		auto accessor = MakeIndexAccessor(std::ref(vec));

		constexpr int idx = 1;
		CHECK(accessor(idx) == 2);
	}

	SECTION("Index with size_t")
	{
		std::vector vec{1, 2, 3};
		auto accessor = MakeIndexAccessor(std::ref(vec));

		constexpr std::size_t idx = 2;
		CHECK(accessor(idx) == 3);
	}

	SECTION("Index with unsigned")
	{
		std::vector vec{1, 2, 3};
		auto accessor = MakeIndexAccessor(std::ref(vec));

		constexpr unsigned idx = 0;
		CHECK(accessor(idx) == 1);
	}
}


TEST_CASE("MakeIndexAccessor - Different value types")
{
	SECTION("std::vector of strings")
	{
		std::vector<std::string> vec{"hello", "world", "test"};

		auto accessor = MakeIndexAccessor(std::ref(vec));

		CHECK(accessor(0) == "hello");
		CHECK(accessor(1) == "world");
		CHECK(accessor(2) == "test");

		accessor(1) = "everyone";
		CHECK(vec[1] == "everyone");
	}

	SECTION("std::vector of doubles")
	{
		std::vector vec{1.1, 2.2, 3.3};

		auto accessor = MakeIndexAccessor(std::ref(vec));

		CHECK(accessor(0) == 1.1);
		CHECK(accessor(1) == 2.2);
		CHECK(accessor(2) == 3.3);
	}

	SECTION("std::array of pairs")
	{
		std::array<std::pair<int, int>, 3> arr{
			std::pair{1, 2},
			std::pair{3, 4},
			std::pair{5, 6}
		};

		auto accessor = MakeIndexAccessor(std::ref(arr));

		CHECK(accessor(0).first == 1);
		CHECK(accessor(0).second == 2);
		CHECK(accessor(1).first == 3);
		CHECK(accessor(2).second == 6);
	}
}


TEST_CASE("MakeIndexAccessor - Move semantics")
{
	SECTION("Takes ownership of move-only types")
	{
		struct MoveOnlyIndexable
		{
			std::vector<int> data;

			explicit MoveOnlyIndexable(std::vector<int> d) : data(std::move(d)) {}
			MoveOnlyIndexable(const MoveOnlyIndexable&) = delete;
			MoveOnlyIndexable(MoveOnlyIndexable&&) noexcept = default;
			MoveOnlyIndexable& operator=(const MoveOnlyIndexable&) = delete;
			MoveOnlyIndexable& operator=(MoveOnlyIndexable&&) noexcept = default;

			int& operator[](const std::size_t i) noexcept { return data[i]; }
			const int& operator[](const std::size_t i) const noexcept { return data[i]; }
		};

		MoveOnlyIndexable movable{std::vector{1, 2, 3, 4, 5}};
		auto accessor = MakeIndexAccessor(std::move(movable));

		CHECK(accessor(0) == 1);
		CHECK(accessor(2) == 3);
		// accessor(2) = 42;  // not supported in current version
		// CHECK(accessor(2) == 42);
	}
}


TEST_CASE("MakeIndexAccessor - Const correctness")
{
	SECTION("Const vector by reference")
	{
		const std::vector vec{1, 2, 3, 4, 5};

		auto accessor = MakeIndexAccessor(std::ref(vec));

		// Should be able to read
		CHECK(accessor(0) == 1);
		CHECK(accessor(2) == 3);

		// The return type should be const int&
		STATIC_CHECK(std::is_same_v<decltype(accessor(0)), const int&>);
	}

	SECTION("Non-const vector by reference")
	{
		std::vector vec{1, 2, 3, 4, 5};

		auto accessor = MakeIndexAccessor(std::ref(vec));

		// Should be able to read and write
		CHECK(accessor(0) == 1);
		accessor(0) = 99;
		CHECK(vec[0] == 99);

		// The return type should be int&
		STATIC_CHECK(std::is_same_v<decltype(accessor(0)), int&>);
	}
}


TEST_CASE("MakeIndexAccessor - noexcept propagation")
{
	{
		struct /* UNNAMED */
		{
			int operator[](int) const
			{
				return 0;
			}
		} s;

		{
			auto accessor = MakeIndexAccessor(s);
			STATIC_CHECK_FALSE(noexcept(accessor(0)));
			(void)accessor;
		}
		{
			auto accessor = MakeIndexAccessor(std::ref(s));
			STATIC_CHECK_FALSE(noexcept(accessor(0)));
			(void)accessor;
		}
	}

	{
		struct /* UNNAMED */
		{
			int operator[](int) const noexcept
			{
				return 0;
			}
		} s;

		{
			auto accessor = MakeIndexAccessor(s);
			STATIC_CHECK(noexcept(accessor(0)));
			(void)accessor;
		}
		{
			auto accessor = MakeIndexAccessor(std::ref(s));
			STATIC_CHECK(noexcept(accessor(0)));
			(void)accessor;
		}
	}
}


TEST_CASE("MakeIndexAccessor - Custom indexable type")
{
	SECTION("Custom class with operator[]")
	{
		struct Matrix
		{
			std::vector<int> data;
			std::size_t cols;

			Matrix(const std::size_t r, const std::size_t c) : data(r * c), cols(c) {}

			int& operator[](const std::size_t i) { return data[i]; }
			const int& operator[](const std::size_t i) const { return data[i]; }
		};

		Matrix mat(2, 3);
		mat[0] = 1;
		mat[1] = 2;
		mat[2] = 3;

		auto accessor = MakeIndexAccessor(std::ref(mat));

		CHECK(accessor(0) == 1);
		CHECK(accessor(1) == 2);
		CHECK(accessor(2) == 3);

		accessor(1) = 99;
		CHECK(mat[1] == 99);
	}

	SECTION("Proxy return type")
	{
		struct ProxyContainer
		{
			mutable int last_accessed = -1;

			struct Proxy
			{
				int& value;
				ProxyContainer* container;
				std::size_t index;

				// ReSharper disable once CppNonExplicitConversionOperator
				operator int() const { return value; }
				Proxy& operator=(const int v)
				{
					value = v;
					container->last_accessed = index;
					return *this;
				}
			};

			std::vector<int> data;

			ProxyContainer(const std::initializer_list<int> init) : data(init) {}

			Proxy operator[](const std::size_t i) { return Proxy{data[i], this, i}; }
			int operator[](const std::size_t i) const { return data[i]; }
		};

		ProxyContainer container{1, 2, 3, 4, 5};

		auto accessor = MakeIndexAccessor(std::ref(container));

		CHECK(accessor(0) == 1);
		CHECK(accessor(2) == 3);

		accessor(2) = 42;
		CHECK(container.data[2] == 42);
		CHECK(container.last_accessed == 2);
	}
}


TEST_CASE("MakeIndexAccessor - Iterator usage")
{
	SECTION("Random access iterator")
	{
		std::vector vec{10, 20, 30, 40, 50};

		auto accessor = MakeIndexAccessor(vec.begin());

		CHECK(accessor(0) == 10);
		CHECK(accessor(1) == 20);
		CHECK(accessor(2) == 30);

		accessor(1) = 99;
		CHECK(vec[1] == 99);
	}

	SECTION("Const iterator")
	{
		const std::vector vec{10, 20, 30, 40, 50};

		auto accessor = MakeIndexAccessor(vec.cbegin());

		CHECK(accessor(0) == 10);
		CHECK(accessor(2) == 30);
	}
}


TEST_CASE("MakeIndexAccessor - Copy and move semantics of accessor")
{
	SECTION("Accessor is copyable")
	{
		std::vector vec{1, 2, 3};
		auto accessor1 = MakeIndexAccessor(std::ref(vec));
		auto accessor2 = accessor1;  // Copy

		CHECK(accessor1(0) == 1);
		CHECK(accessor2(0) == 1);

		accessor2(0) = 99;
		CHECK(vec[0] == 99);
		CHECK(accessor1(0) == 99);
	}

	SECTION("Accessor is movable")
	{
		std::vector vec{1, 2, 3};
		auto accessor1 = MakeIndexAccessor(std::ref(vec));
		auto accessor2 = std::move(accessor1);  // Move

		CHECK(accessor2(0) == 1);
		accessor2(0) = 99;
		CHECK(vec[0] == 99);
	}
}


TEST_CASE("MakeIndexAccessor - Multiple accessors")
{
	SECTION("Multiple accessors to same container")
	{
		std::vector vec{1, 2, 3, 4, 5};

		auto acc1 = MakeIndexAccessor(std::ref(vec));
		auto acc2 = MakeIndexAccessor(std::ref(vec));

		CHECK(acc1(0) == 1);
		CHECK(acc2(0) == 1);

		acc1(0) = 10;
		CHECK(acc2(0) == 10);

		acc2(1) = 20;
		CHECK(acc1(1) == 20);
	}
}
