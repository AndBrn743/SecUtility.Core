//
// Created by Andy on 6/10/2026.
//
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <SecUtility/Collection/MultidimensionalArray.hpp>

#include <algorithm>
#include <numeric>
#include <type_traits>

using SecUtility::MultidimensionalArray;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Fill arr(i,j,...) with its flat index value.
template <typename Arr>
void iota_fill(Arr& arr)
{
	int n = 0;
	for (auto& v : arr)
	{
		v = static_cast<typename Arr::value_type>(n++);
	}
}

TEST_CASE("MultidimensionalArray")
{
	// =============================================================================
	// 1. Compile-time properties
	// =============================================================================

	SECTION("Compile-time properties", "[static]")
	{

		SECTION("rank")
		{
			STATIC_CHECK(MultidimensionalArray<int, 5>::Rank == 1);
			STATIC_CHECK(MultidimensionalArray<int, 3, 4>::Rank == 2);
			STATIC_CHECK(MultidimensionalArray<int, 2, 3, 4>::Rank == 3);
			STATIC_CHECK(MultidimensionalArray<int, 2, 3, 4, 5>::Rank == 4);
		}

		SECTION("total_size")
		{
			STATIC_CHECK(MultidimensionalArray<int, 7>::TotalSize == 7);
			STATIC_CHECK(MultidimensionalArray<int, 3, 4>::TotalSize == 12);
			STATIC_CHECK(MultidimensionalArray<int, 2, 3, 4>::TotalSize == 24);
			STATIC_CHECK(MultidimensionalArray<int, 2, 3, 4, 5>::TotalSize == 120);
		}

		SECTION("shape")
		{
			using A = MultidimensionalArray<int, 2, 3, 4>;
			STATIC_CHECK(A::Shape[0] == 2);
			STATIC_CHECK(A::Shape[1] == 3);
			STATIC_CHECK(A::Shape[2] == 4);
		}

		SECTION("row-major strides")
		{
			// shape {2,3,4} → strides {12, 4, 1}
			using A = MultidimensionalArray<int, 2, 3, 4>;
			STATIC_CHECK(A::Strides[0] == 12);
			STATIC_CHECK(A::Strides[1] == 4);
			STATIC_CHECK(A::Strides[2] == 1);

			// shape {5,6} → strides {6, 1}
			using B = MultidimensionalArray<int, 5, 6>;
			STATIC_CHECK(B::Strides[0] == 6);
			STATIC_CHECK(B::Strides[1] == 1);

			// 1-D: stride is always 1
			using C = MultidimensionalArray<int, 9>;
			STATIC_CHECK(C::Strides[0] == 1);
		}

		SECTION("type aliases")
		{
			using A = MultidimensionalArray<double, 3, 3>;
			STATIC_CHECK(std::is_same_v<typename A::value_type, double>);
			STATIC_CHECK(std::is_same_v<typename A::reference, double&>);
			STATIC_CHECK(std::is_same_v<typename A::const_reference, const double&>);
			STATIC_CHECK(std::is_same_v<typename A::pointer, double*>);
		}
	}


	// =============================================================================
	// 2. Construction
	// =============================================================================

	SECTION("Default construction zero-initialises", "[construction]")
	{
		MultidimensionalArray<int, 2, 3, 4> arr;
		for (const auto& v : arr)
		{
			REQUIRE(v == 0);
		}
	}

	SECTION("Fill construction", "[construction]")
	{
		MultidimensionalArray<float, 3, 3> arr(3.14f);
		for (const auto& v : arr)
		{
			REQUIRE(v == Catch::Approx(3.14f));
		}
	}

	SECTION("Initialiser-list construction", "[construction]")
	{
		MultidimensionalArray<int, 2, 3> arr{0, 1, 2, 3, 4, 5};
		int expected = 0;
		for (const auto& v : arr)
		{
			REQUIRE(v == expected++);
		}
	}

	SECTION("Initialiser-list size mismatch throws", "[construction]")
	{
		REQUIRE_THROWS_AS((MultidimensionalArray<int, 2, 3>{0, 1, 2}), SecUtility::InvalidArgumentException);
	}

	SECTION("from_values factory (constexpr)", "[construction]")
	{
		constexpr auto arr = MultidimensionalArray<int, 2, 3>::FromValues(10, 11, 12, 13, 14, 15);
		STATIC_CHECK(arr(0, 0) == 10);
		STATIC_CHECK(arr(0, 2) == 12);
		STATIC_CHECK(arr(1, 2) == 15);
		REQUIRE(arr(1, 1) == 14);
	}

	SECTION("Copy construction produces independent copy", "[construction]")
	{
		MultidimensionalArray<int, 3> a{1, 2, 3};
		auto b = a;
		b(0) = 99;
		REQUIRE(a(0) == 1);  // original unchanged
		REQUIRE(b(0) == 99);
	}

	SECTION("Copy assignment", "[construction]")
	{
		MultidimensionalArray<int, 3> a{1, 2, 3};
		MultidimensionalArray<int, 3> b;
		b = a;
		REQUIRE(b(0) == 1);
		REQUIRE(b(2) == 3);
		b(0) = 99;
		REQUIRE(a(0) == 1);  // original unchanged
	}

	SECTION("Move construction", "[construction]")
	{
		MultidimensionalArray<int, 3> a{7, 8, 9};
		auto b = std::move(a);
		REQUIRE(b(0) == 7);
		REQUIRE(b(2) == 9);
	}


	// =============================================================================
	// 3. Element access — correctness
	// =============================================================================

	SECTION("1-D read/write", "[access][1d]")
	{
		MultidimensionalArray<int, 6> arr;
		iota_fill(arr);
		for (std::size_t i = 0; i < 6; ++i)
		{
			REQUIRE(arr(i) == static_cast<int>(i));
		}
		arr(3) = 999;
		REQUIRE(arr(3) == 999);
	}

	SECTION("2-D read/write — row-major layout", "[access][2d]")
	{
		MultidimensionalArray<int, 4, 5> m;
		iota_fill(m);
		// flat index for (r,c) = r*5 + c
		for (std::size_t r = 0; r < 4; ++r)
		{
			for (std::size_t c = 0; c < 5; ++c)
			{
				REQUIRE(m(r, c) == static_cast<int>(r * 5 + c));
			}
		}
	}

	SECTION("3-D read/write — row-major layout", "[access][3d]")
	{
		MultidimensionalArray<int, 2, 3, 4> t;
		iota_fill(t);
		for (std::size_t i = 0; i < 2; ++i)
		{
			for (std::size_t j = 0; j < 3; ++j)
			{
				for (std::size_t k = 0; k < 4; ++k)
				{
					REQUIRE(t(i, j, k) == static_cast<int>(i * 12 + j * 4 + k));
				}
			}
		}
	}

	SECTION("4-D read/write", "[access][4d]")
	{
		MultidimensionalArray<int, 2, 3, 4, 5> q;
		iota_fill(q);
		REQUIRE(q(0, 0, 0, 0) == 0);
		REQUIRE(q(1, 2, 3, 4) == static_cast<int>(q.TotalSize - 1));
		// spot-check: (1,0,0,0) = 1*60
		REQUIRE(q(1, 0, 0, 0) == 60);
	}

	SECTION("const access via const reference", "[access]")
	{
		const MultidimensionalArray<int, 2, 3> cm{0, 1, 2, 3, 4, 5};
		REQUIRE(cm(0, 0) == 0);
		REQUIRE(cm(1, 2) == 5);
	}

	SECTION("flat() access matches operator()", "[access]")
	{
		MultidimensionalArray<int, 3, 4> m;
		iota_fill(m);
		for (std::size_t i = 0; i < m.TotalSize; ++i)
		{
			REQUIRE(m.Flat(i) == static_cast<int>(i));
		}
	}

	SECTION("data() pointer arithmetic matches operator()", "[access]")
	{
		MultidimensionalArray<int, 3, 4> m;
		iota_fill(m);
		const int* ptr = m.data();
		for (std::size_t r = 0; r < 3; ++r)
		{
			for (std::size_t c = 0; c < 4; ++c)
			{
				REQUIRE(ptr[r * 4 + c] == m(r, c));
			}
		}
	}


	// =============================================================================
	// 4. Bounds checking
	// =============================================================================

	// SECTION("operator() throws out_of_range for out-of-bounds index", "[bounds]")
	// {
	// 	MultidimensionalArray<int, 3, 4> m;
	//
	// 	SECTION("first dimension overflow")
	// 	{
	// 		REQUIRE_THROWS_AS(m(3, 0), std::out_of_range);
	// 	}
	// 	SECTION("second dimension overflow")
	// 	{
	// 		REQUIRE_THROWS_AS(m(0, 4), std::out_of_range);
	// 	}
	// 	SECTION("both dimensions overflow")
	// 	{
	// 		REQUIRE_THROWS_AS(m(10, 10), std::out_of_range);
	// 	}
	// }

	SECTION("at() throws out_of_range", "[bounds]")
	{
		MultidimensionalArray<int, 2, 2> m;
		REQUIRE_THROWS_AS(m.At(2, 0), SecUtility::ArgumentOutOfRangeException);
		REQUIRE_THROWS_AS(m.At(0, 2), SecUtility::ArgumentOutOfRangeException);
	}

	SECTION("Valid boundary indices do NOT throw", "[bounds]")
	{
		MultidimensionalArray<int, 3, 4> m;
		REQUIRE_NOTHROW(m(0, 0));
		REQUIRE_NOTHROW(m(2, 3));  // last valid indices
	}


	// =============================================================================
	// 5. Observers
	// =============================================================================

	SECTION("size() returns total element count", "[observers]")
	{
		REQUIRE(MultidimensionalArray<int, 5>().Size() == 5u);
		REQUIRE((MultidimensionalArray<int, 3, 4>().Size()) == 12u);
		REQUIRE((MultidimensionalArray<int, 2, 3, 4>().Size()) == 24u);
	}

	SECTION("extent() returns per-dimension size", "[observers]")
	{
		MultidimensionalArray<int, 2, 3, 4> arr;
		REQUIRE(arr.Extent(0) == 2u);
		REQUIRE(arr.Extent(1) == 3u);
		REQUIRE(arr.Extent(2) == 4u);
	}

	SECTION("IsEmpty() is false for non-zero arrays", "[observers]")
	{
		REQUIRE_FALSE(MultidimensionalArray<int, 1>().IsEmpty());
		REQUIRE_FALSE((MultidimensionalArray<int, 2, 3>().IsEmpty()));
	}


	// =============================================================================
	// 6. Iterators
	// =============================================================================

	SECTION("Flat iteration traverses elements in row-major order", "[iterators]")
	{
		MultidimensionalArray<int, 3, 4> m;
		iota_fill(m);

		int expected = 0;
		for (auto it = m.begin(); it != m.end(); ++it)
		{
			REQUIRE(*it == expected++);
		}
		REQUIRE(expected == 12);
	}

	SECTION("Range-for works", "[iterators]")
	{
		MultidimensionalArray<int, 5> arr{10, 20, 30, 40, 50};
		int sum = 0;
		for (int v : arr)
		{
			sum += v;
		}
		REQUIRE(sum == 150);
	}

	SECTION("cbegin/cend give const iterators", "[iterators]")
	{
		MultidimensionalArray<int, 4> arr{1, 2, 3, 4};
		auto it = arr.cbegin();
		STATIC_CHECK(std::is_same_v<decltype(it), MultidimensionalArray<int, 4>::const_iterator>);
		REQUIRE(*it == 1);
	}

	SECTION("Iterator distance equals total_size", "[iterators]")
	{
		MultidimensionalArray<int, 2, 3, 4> arr;
		REQUIRE(std::distance(arr.begin(), arr.end()) == static_cast<std::ptrdiff_t>(arr.TotalSize));
	}

	SECTION("std::sort works on the flat range", "[iterators]")
	{
		MultidimensionalArray<int, 2, 3> arr{5, 3, 1, 4, 2, 0};
		std::sort(arr.begin(), arr.end());
		for (int i = 0; i < 6; ++i)
		{
			REQUIRE(arr.Flat(i) == i);
		}
	}

	SECTION("std::accumulate sums all elements", "[iterators]")
	{
		MultidimensionalArray<int, 3, 3> arr;
		iota_fill(arr);  // 0..8
		int sum = std::accumulate(arr.begin(), arr.end(), 0);
		REQUIRE(sum == 36);
	}


	// =============================================================================
	// 7. fill() and for_each()
	// =============================================================================

	SECTION("fill() sets all elements", "[mutation]")
	{
		MultidimensionalArray<int, 3, 4> m;
		m.Fill(42);
		for (const auto& v : m)
		{
			REQUIRE(v == 42);
		}
	}

	SECTION("for_each() mutable — doubles every element", "[mutation]")
	{
		MultidimensionalArray<int, 2, 3> arr{1, 2, 3, 4, 5, 6};
		arr.Foreach([](int& v) { v *= 2; });
		int expected = 2;
		for (const auto& v : arr)
		{
			REQUIRE(v == expected);
			expected += 2;
		}
	}

	SECTION("for_each() const — accumulates without modifying", "[mutation]")
	{
		const MultidimensionalArray<int, 4> arr{10, 20, 30, 40};
		int sum = 0;
		arr.Foreach([&sum](const int& v) { sum += v; });
		REQUIRE(sum == 100);
	}


	// =============================================================================
	// 8. Equality operators
	// =============================================================================

	SECTION("Equality and inequality", "[equality]")
	{
		MultidimensionalArray<int, 2, 3> a{0, 1, 2, 3, 4, 5};
		MultidimensionalArray<int, 2, 3> b{0, 1, 2, 3, 4, 5};
		MultidimensionalArray<int, 2, 3> c{0, 1, 2, 3, 4, 99};

		REQUIRE(a == b);
		REQUIRE_FALSE(a != b);
		REQUIRE(a != c);
		REQUIRE_FALSE(a == c);
	}

	SECTION("Self-equality", "[equality]")
	{
		MultidimensionalArray<int, 3> arr{1, 2, 3};
		REQUIRE(arr == arr);
	}

	SECTION("Default-constructed arrays are equal", "[equality]")
	{
		MultidimensionalArray<int, 2, 3> a, b;
		REQUIRE(a == b);
	}


#if false
	// =============================================================================
	// 9. Slice
	// =============================================================================

	SECTION("slice() on 3-D: reduces to 2-D", "[slice]")
	{
		MultidimensionalArray<int, 2, 3, 4> arr;
		iota_fill(arr);

		auto s0 = arr.Slice<1>(0);
		auto s1 = arr.Slice<1>(1);

		SECTION("sub-array type has correct rank and shape")
		{
			STATIC_CHECK(decltype(s0)::Rank == 2);
			STATIC_CHECK(decltype(s0)::Shape[0] == 3);
			STATIC_CHECK(decltype(s0)::Shape[1] == 4);
		}

		SECTION("slice 0 matches first 12 elements")
		{
			for (std::size_t j = 0; j < 3; ++j)
			{
				for (std::size_t k = 0; k < 4; ++k)
				{
					REQUIRE(s0(j, k) == arr(0, j, k));
				}
			}
		}

		SECTION("slice 1 matches second 12 elements")
		{
			for (std::size_t j = 0; j < 3; ++j)
			{
				for (std::size_t k = 0; k < 4; ++k)
				{
					REQUIRE(s1(j, k) == arr(1, j, k));
				}
			}
		}

		SECTION("slice is a copy — mutating it does not affect original")
		{
			s0(0, 0) = 9999;
			REQUIRE(arr(0, 0, 0) == 0);
		}
	}

	SECTION("slice() on 2-D: reduces to 1-D", "[slice]")
	{
		MultidimensionalArray<int, 4, 5> m;
		iota_fill(m);

		auto row2 = m.Slice<1>(2);
		STATIC_CHECK(decltype(row2)::Rank == 1);
		STATIC_CHECK(decltype(row2)::TotalSize == 5);

		for (std::size_t c = 0; c < 5; ++c)
		{
			REQUIRE(row2(c) == m(2, c));
		}
	}

	SECTION("slice() on 4-D: reduces to 3-D", "[slice]")
	{
		MultidimensionalArray<int, 2, 3, 4, 5> arr;
		iota_fill(arr);

		auto s = arr.Slice<1>(1);
		STATIC_CHECK(decltype(s)::Rank == 3);
		STATIC_CHECK(decltype(s)::Shape[0] == 3);

		for (std::size_t j = 0; j < 3; ++j)
		{
			for (std::size_t k = 0; k < 4; ++k)
			{
				for (std::size_t l = 0; l < 5; ++l)
				{
					REQUIRE(s(j, k, l) == arr(1, j, k, l));
				}
			}
		}
	}
#endif


	SECTION("Works with non-trivial types (std::string)", "[types]")
	{
		MultidimensionalArray<std::string, 2, 3> arr;
		arr(0, 0) = "hello";
		arr(1, 2) = "world";
		REQUIRE(arr(0, 0) == "hello");
		REQUIRE(arr(1, 2) == "world");
		REQUIRE(arr(0, 1).empty());  // default-constructed string
	}


	// =============================================================================
	// 11. Edge cases & corner cases
	// =============================================================================

	SECTION("1*1 array", "[edge]")
	{
		MultidimensionalArray<int, 1> arr;
		arr(0) = 42;
		REQUIRE(arr(0) == 42);
		REQUIRE(arr.Size() == 1u);
	}

	SECTION("1*1*1 array", "[edge]")
	{
		MultidimensionalArray<int, 1, 1, 1> arr;
		arr(0, 0, 0) = 7;
		REQUIRE(arr(0, 0, 0) == 7);
		REQUIRE(arr.Size() == 1u);
	}

	SECTION("Large flat array (1-D of 10000)", "[edge]")
	{
		MultidimensionalArray<int, 10000> arr;
		iota_fill(arr);
		REQUIRE(arr(0) == 0);
		REQUIRE(arr(9999) == 9999);
		REQUIRE(arr(5000) == 5000);
	}

	SECTION("Index at last valid position does not throw", "[edge]")
	{
		MultidimensionalArray<int, 3, 4, 5> arr;
		REQUIRE_NOTHROW(arr(2, 3, 4));
	}

	SECTION("Index one past last position throws", "[edge]")
	{
		MultidimensionalArray<int, 3, 4, 5> arr;
		REQUIRE_THROWS_AS(arr.At(3, 0, 0), SecUtility::ArgumentOutOfRangeException);
		REQUIRE_THROWS_AS(arr.At(0, 4, 0), SecUtility::ArgumentOutOfRangeException);
		REQUIRE_THROWS_AS(arr.At(0, 0, 5), SecUtility::ArgumentOutOfRangeException);
	}

	SECTION("Fill then overwrite a single element", "[edge]")
	{
		MultidimensionalArray<int, 3, 3> arr(1);
		arr(1, 1) = 99;
		for (std::size_t i = 0; i < 3; ++i)
		{
			for (std::size_t j = 0; j < 3; ++j)
			{
				REQUIRE(arr(i, j) == (i == 1 && j == 1 ? 99 : 1));
			}
		}
	}

	SECTION("float values preserve precision through round-trip", "[edge]")
	{
		constexpr float pi = 3.14159265358979f;
		MultidimensionalArray<float, 2, 2> arr;
		arr(0, 0) = pi;
		REQUIRE(arr(0, 0) == Catch::Approx(pi));
	}

	SECTION("Negative integer values stored correctly", "[edge]")
	{
		MultidimensionalArray<int, 3> arr{-1, -100, -32768};
		REQUIRE(arr(0) == -1);
		REQUIRE(arr(1) == -100);
		REQUIRE(arr(2) == -32768);
	}


	// =============================================================================
	// 12. STL algorithm interoperability
	// =============================================================================

	SECTION("std::fill works via iterators", "[stl]")
	{
		MultidimensionalArray<int, 3, 3> arr;
		std::fill(arr.begin(), arr.end(), 7);
		for (const auto& v : arr)
		{
			REQUIRE(v == 7);
		}
	}

	SECTION("std::count works via iterators", "[stl]")
	{
		MultidimensionalArray<int, 2, 3> arr{1, 2, 1, 1, 2, 3};
		REQUIRE(std::count(arr.begin(), arr.end(), 1) == 3);
		REQUIRE(std::count(arr.begin(), arr.end(), 2) == 2);
	}

	SECTION("std::find works via iterators", "[stl]")
	{
		MultidimensionalArray<int, 5> arr{10, 20, 30, 40, 50};
		auto it = std::find(arr.begin(), arr.end(), 30);
		REQUIRE(it != arr.end());
		REQUIRE(*it == 30);
		REQUIRE(std::distance(arr.begin(), it) == 2);
	}

	SECTION("std::copy round-trips through raw pointer", "[stl]")
	{
		MultidimensionalArray<int, 2, 3> src{0, 1, 2, 3, 4, 5};
		MultidimensionalArray<int, 2, 3> dst;
		std::copy(src.begin(), src.end(), dst.begin());
		REQUIRE(src == dst);
	}

	SECTION("std::transform applies element-wise operation", "[stl]")
	{
		MultidimensionalArray<int, 4> a{1, 2, 3, 4};
		MultidimensionalArray<int, 4> b{10, 20, 30, 40};
		MultidimensionalArray<int, 4> result;
		std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<int>{});
		MultidimensionalArray<int, 4> expected{11, 22, 33, 44};
		REQUIRE(result == expected);
	}

	SECTION("std::reverse reverses flat storage", "[stl]")
	{
		MultidimensionalArray<int, 2, 3> arr{0, 1, 2, 3, 4, 5};
		std::reverse(arr.begin(), arr.end());
		MultidimensionalArray<int, 2, 3> expected{5, 4, 3, 2, 1, 0};
		REQUIRE(arr == expected);
	}

	SECTION("std::max_element / min_element work", "[stl]")
	{
		MultidimensionalArray<int, 3, 3> arr{5, 2, 8, 1, 9, 3, 7, 4, 6};
		REQUIRE(*std::max_element(arr.begin(), arr.end()) == 9);
		REQUIRE(*std::min_element(arr.begin(), arr.end()) == 1);
	}
}

// =============================================================================
// 10. Type template tests — behaviour consistent across value types
// =============================================================================

TEMPLATE_TEST_CASE("MultidimensionalArray: Default construction zero/value initialises",
                   "[types]",
                   int,
                   float,
                   double,
                   unsigned,
                   short,
                   long long)
{
	MultidimensionalArray<TestType, 3, 3> arr;
	for (const auto& v : arr)
	{
		REQUIRE(v == TestType{0});
	}
}

TEMPLATE_TEST_CASE("MultidimensionalArray: Read/write round-trips", "[types]", int, float, double, unsigned)
{
	MultidimensionalArray<TestType, 4, 4> arr;
	for (std::size_t i = 0; i < 4; ++i)
	{
		for (std::size_t j = 0; j < 4; ++j)
		{
			arr(i, j) = static_cast<TestType>(i * 4 + j);
		}
	}

	for (std::size_t i = 0; i < 4; ++i)
	{
		for (std::size_t j = 0; j < 4; ++j)
		{
			REQUIRE(arr(i, j) == static_cast<TestType>(i * 4 + j));
		}
	}
}