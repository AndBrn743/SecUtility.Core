//
// Created by Andy on 5/20/2026.
//

#include <SecUtility/Collection/KeyedArray.hpp>

#include <cassert>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <string>
#include <vector>

#if __cplusplus >= 202002L
struct Point
{
	int X;
	int Y;
	constexpr bool operator==(const Point&) const = default;
};

struct Channel
{
	int device;
	int index;
	constexpr bool operator==(const Channel&) const = default;
};

using namespace SecUtility;
#endif

TEST_CASE("KeyedArray")
{
#if __cplusplus < 202002L
	SKIP("Test requires C++20");
#else
	SECTION("basic Get<K>()")
	{
		KeyedArray<int, Point{1, 0}, Point{0, 1}> arr;

		arr.Get<Point{1, 0}>() = 42;
		arr.Get<Point{0, 1}>() = 99;

		CHECK(arr.Get<Point{1, 0}>() == 42);
		CHECK(arr.Get<Point{0, 1}>() == 99);
	}

	SECTION("key order is part of the type")
	{
		KeyedArray<int, Point{1, 0}, Point{0, 1}> ab;
		KeyedArray<int, Point{0, 1}, Point{1, 0}> ba;

		ab.Get<Point{1, 0}>() = 1;
		ab.Get<Point{0, 1}>() = 2;

		ba.Get<Point{0, 1}>() = 10;
		ba.Get<Point{1, 0}>() = 20;

		// Raw storage reflects declaration order, not key identity.
		CHECK(ab.data()[0] == 1);  // Point{1,0} is slot 0 in `ab`
		CHECK(ab.data()[1] == 2);
		CHECK(ba.data()[0] == 10);  // Point{0,1} is slot 0 in `ba`
		CHECK(ba.data()[1] == 20);
	}

	SECTION("runtime operator[]")
	{
		KeyedArray<double, Point{3, 4}, Point{0, 0}, Point{1, 2}> arr;

		arr.Get<Point{3, 4}>() = 1.1;
		arr.Get<Point{0, 0}>() = 2.2;
		arr.Get<Point{1, 2}>() = 3.3;

		Point rt{0, 0};
		CHECK(arr[rt] == 2.2);

		CHECK_THROWS(arr[Point{9, 9}]);
	}

	SECTION("Foreach mutable + const")
	{
		KeyedArray<int, Point{0, 0}, Point{1, 0}, Point{0, 1}> arr;
		arr.Get<Point{0, 0}>() = 10;
		arr.Get<Point{1, 0}>() = 20;
		arr.Get<Point{0, 1}>() = 30;

		// Mutable Foreach - double every value.
		arr.Foreach([](const Point&, int& v) { v *= 2; });
		CHECK(arr.Get<Point{0, 0}>() == 20);
		CHECK(arr.Get<Point{1, 0}>() == 40);
		CHECK(arr.Get<Point{0, 1}>() == 60);

		// Const Foreach - sum values.
		int sum = 0;
		const auto& carr = arr;
		carr.Foreach([&](const Point&, const int& v) { sum += v; });
		CHECK(sum == 120);
	}

	SECTION("Foreach visits in declaration order")
	{
		// Declare in a specific order and confirm iteration matches.
		KeyedArray<int, Point{2, 0}, Point{0, 2}, Point{1, 1}> arr;
		arr.Get<Point{2, 0}>() = 1;
		arr.Get<Point{0, 2}>() = 2;
		arr.Get<Point{1, 1}>() = 3;

		std::vector<Point> visited_keys;
		std::vector<int> visited_vals;
		arr.Foreach(
		        [&](const Point& k, const int& v)
		        {
			        visited_keys.push_back(k);
			        visited_vals.push_back(v);
		        });

		CHECK(visited_keys[0] == Point{2, 0});
		CHECK(visited_keys[1] == Point{0, 2});
		CHECK(visited_keys[2] == Point{1, 1});
		CHECK(visited_vals[0] == 1);
		CHECK(visited_vals[1] == 2);
		CHECK(visited_vals[2] == 3);
	}

	SECTION("Contains<K>")
	{
		using A = KeyedArray<int, Point{1, 0}, Point{0, 1}>;

		CHECK(A::Contains<Point{1, 0}>);
		CHECK(A::Contains<Point{0, 1}>);
		CHECK(!A::Contains<Point{9, 9}>);
	}

	SECTION("static KeySet")
	{
		using A = KeyedArray<int, Point{5, 0}, Point{3, 3}, Point{0, 5}>;

		CHECK(A::KeySet[0] == Point{5, 0});
		CHECK(A::KeySet[1] == Point{3, 3});
		CHECK(A::KeySet[2] == Point{0, 5});
		CHECK(A::Size() == 3);
	}

	SECTION("Channel 'enum+' use case")
	{
		constexpr Channel ADC0{0, 0}, ADC1{0, 1}, DAC0{1, 0};

		KeyedArray<double, ADC0, ADC1, DAC0> readings;
		readings.Get<ADC0>() = 3.14;
		readings.Get<ADC1>() = 2.71;
		readings.Get<DAC0>() = 1.41;

		CHECK(readings.Get<ADC0>() == 3.14);
		CHECK(readings.Get<ADC1>() == 2.71);
		CHECK(readings.Get<DAC0>() == 1.41);
	}

	SECTION("fully constexpr")
	{
		constexpr auto make = []() constexpr
		{
			KeyedArray<int, Point{0, 0}, Point{1, 1}> a;
			a.Get<Point{0, 0}>() = 7;
			a.Get<Point{1, 1}>() = 13;
			return a;
		};

		constexpr auto arr = make();
		STATIC_CHECK(arr.Get<Point{0, 0}>() == 7);
		STATIC_CHECK(arr.Get<Point{1, 1}>() == 13);
	}

	SECTION("operator==")
	{
		KeyedArray<int, Point{0, 0}, Point{1, 1}> a, b;
		a.Get<Point{0, 0}>() = 1;
		b.Get<Point{0, 0}>() = 1;
		a.Get<Point{1, 1}>() = 2;
		b.Get<Point{1, 1}>() = 2;
		CHECK(a == b);

		b.Get<Point{1, 1}>() = 99;
		CHECK(a != b);
	}

	SECTION("fill constructor")
	{
		KeyedArray<int, 1, 2, 3> arr{42};

		CHECK(arr.Get<1>() == 42);
		CHECK(arr.Get<2>() == 42);
		CHECK(arr.Get<3>() == 42);
	}

	SECTION("fill constructor with struct keys")
	{
		KeyedArray<int, Point{0, 0}, Point{1, 1}> arr{-1};

		CHECK(arr.Get<Point{0, 0}>() == -1);
		CHECK(arr.Get<Point{1, 1}>() == -1);
	}

	SECTION("fill constructor is constexpr")
	{
		constexpr auto make = []() constexpr
		{
			KeyedArray<int, Point{0, 0}, Point{1, 1}> arr{7};
			return arr;
		};

		constexpr auto arr = make();

		STATIC_CHECK(arr.Get<Point{0, 0}>() == 7);
		STATIC_CHECK(arr.Get<Point{1, 1}>() == 7);
	}

	SECTION("Fill")
	{
		KeyedArray<int, 1, 2, 3> arr{10, 20, 30};

		arr.Fill(0);

		CHECK(arr.Get<1>() == 0);
		CHECK(arr.Get<2>() == 0);
		CHECK(arr.Get<3>() == 0);
	}

	SECTION("Fill resets after mutation")
	{
		KeyedArray<int, 1, 2> arr{0};

		arr.Get<1>() = 5;
		arr.Get<2>() = 10;

		arr.Fill(99);

		CHECK(arr.Get<1>() == 99);
		CHECK(arr.Get<2>() == 99);
	}

	SECTION("move construction", "[KeyedArray][Phase1]")
	{
		SECTION("Move preserves values - basic test")
		{
			KeyedArray<int, 1, 2, 3> original{10, 20, 30};
			KeyedArray<int, 1, 2, 3> moved{std::move(original)};

			REQUIRE(moved.data()[0] == 10);
			REQUIRE(moved.data()[1] == 20);
			REQUIRE(moved.data()[2] == 30);
		}

		SECTION("Move construction with move-only type (std::unique_ptr)")
		{
			// Create unique_ptrs and capture raw pointers
			auto p1 = std::make_unique<int>(42);
			auto p2 = std::make_unique<int>(99);
			int* raw1 = p1.get();
			int* raw2 = p2.get();

			// Move into KeyedArray
			KeyedArray<std::unique_ptr<int>, 0, 1> ka{std::move(p1), std::move(p2)};

			// Verify source pointers are now null (moved)
			REQUIRE(p1.get() == nullptr);
			REQUIRE(p2.get() == nullptr);

			// Verify KeyedArray owns the pointers
			REQUIRE(ka.data()[0].get() == raw1);
			REQUIRE(ka.data()[1].get() == raw2);
			REQUIRE(*ka.data()[0] == 42);
			REQUIRE(*ka.data()[1] == 99);
		}

		SECTION("Move KeyedArray transfers ownership of unique_ptrs")
		{
			// Create and move into first KeyedArray
			auto p1 = std::make_unique<int>(100);
			auto p2 = std::make_unique<int>(200);
			int* raw1 = p1.get();
			int* raw2 = p2.get();

			KeyedArray<std::unique_ptr<int>, 0, 1> ka1{std::move(p1), std::move(p2)};
			REQUIRE(ka1.data()[0].get() == raw1);
			REQUIRE(ka1.data()[1].get() == raw2);

			// Move KeyedArray to another
			KeyedArray<std::unique_ptr<int>, 0, 1> ka2{std::move(ka1)};

			// Verify ka1 no longer owns the pointers
			REQUIRE(ka1.data()[0].get() == nullptr);
			REQUIRE(ka1.data()[1].get() == nullptr);

			// Verify ka2 owns the original pointers
			REQUIRE(ka2.data()[0].get() == raw1);
			REQUIRE(ka2.data()[1].get() == raw2);
			REQUIRE(*ka2.data()[0] == 100);
			REQUIRE(*ka2.data()[1] == 200);
		}
	}
#endif
}

TEST_CASE("Concat(KeyedArray...)")
{
#if __cplusplus < 202002L
	SKIP("Test requires C++20");
#else
	SECTION("basic merge of two arrays")
	{
		SECTION("Concat two int arrays")
		{
			KeyedArray<int, 1, 2> a1{10, 20};
			KeyedArray<int, 3, 4> a2{30, 40};

			auto merged = Concat(a1, a2);

			// Check all keys are present
			CHECK(merged.Get<1>() == 10);
			CHECK(merged.Get<2>() == 20);
			CHECK(merged.Get<3>() == 30);
			CHECK(merged.Get<4>() == 40);

			// Check size
			CHECK(merged.Size() == 4);

			// Check key order is preserved
			CHECK(merged.KeySet[0] == 1);
			CHECK(merged.KeySet[1] == 2);
			CHECK(merged.KeySet[2] == 3);
			CHECK(merged.KeySet[3] == 4);
		}

		SECTION("Concat with struct keys")
		{
			KeyedArray<int, Point{1, 0}, Point{0, 1}> a1{100, 200};
			KeyedArray<int, Point{2, 0}, Point{0, 2}> a2{300, 400};

			auto merged = Concat(a1, a2);

			CHECK(merged.Get<Point{1, 0}>() == 100);
			CHECK(merged.Get<Point{0, 1}>() == 200);
			CHECK(merged.Get<Point{2, 0}>() == 300);
			CHECK(merged.Get<Point{0, 2}>() == 400);
			CHECK(merged.Size() == 4);
		}
	}

	SECTION("type conversion")
	{
		SECTION("Merge int and double arrays")
		{
			KeyedArray<int, 1, 2> a1{10, 20};
			KeyedArray<double, 3, 4> a2{3.14, 2.71};

			auto merged = Concat(a1, a2);

			// Result type should be double (common_type of int and double)
			static_assert(std::is_same_v<decltype(merged)::ValueType, double>);

			// Values are properly converted
			CHECK(merged.Get<1>() == 10.0);
			CHECK(merged.Get<2>() == 20.0);
			CHECK(merged.Get<3>() == 3.14);
			CHECK(merged.Get<4>() == 2.71);
		}

		SECTION("Merge float and int arrays")
		{
			KeyedArray<float, 0> a1{1.5f};
			KeyedArray<int, 1> a2{2};

			auto merged = Concat(a1, a2);

			// Result type should be float (common_type of float and int)
			static_assert(std::is_same_v<decltype(merged)::ValueType, float>);

			CHECK(merged.Get<0>() == 1.5f);
			CHECK(merged.Get<1>() == 2.0f);
		}
	}

	SECTION("move semantics")
	{
		SECTION("Concat with rvalue moves unique_ptrs")
		{
			auto p1 = std::make_unique<int>(42);
			auto p2 = std::make_unique<int>(99);
			auto p3 = std::make_unique<int>(100);
			auto p4 = std::make_unique<int>(200);

			int* raw1 = p1.get();
			int* raw2 = p2.get();
			int* raw3 = p3.get();
			int* raw4 = p4.get();

			KeyedArray<std::unique_ptr<int>, 0, 1> a1{std::move(p1), std::move(p2)};
			KeyedArray<std::unique_ptr<int>, 2, 3> a2{std::move(p3), std::move(p4)};

			// Merge using move
			auto merged = Concat(std::move(a1), std::move(a2));

			// Verify ownership transferred to merged array
			CHECK(merged.data()[0].get() == raw1);
			CHECK(merged.data()[1].get() == raw2);
			CHECK(merged.data()[2].get() == raw3);
			CHECK(merged.data()[3].get() == raw4);
			CHECK(*merged.data()[0] == 42);
			CHECK(*merged.data()[1] == 99);
			CHECK(*merged.data()[2] == 100);
			CHECK(*merged.data()[3] == 200);

			// Verify source arrays no longer own the data
			CHECK(a1.data()[0].get() == nullptr);
			CHECK(a1.data()[1].get() == nullptr);
			CHECK(a2.data()[0].get() == nullptr);
			CHECK(a2.data()[1].get() == nullptr);
		}


		SECTION("Concat with mixed move and copy")
		{
			// Long strings to avoid small string optimization
			std::string s1 = "This is a very long string to avoid SSO forty two";
			std::string s2 = "This is a very long string to avoid SSO ninety nine";
			std::string s1_copy = s1;
			std::string s2_copy = s2;

			KeyedArray<std::string, 0, 1> a1{std::move(s1), std::move(s2)};
			KeyedArray<std::string, 2, 3> a2{"thirty", "forty"};

			// a1 is moved, a2 is copied
			auto merged = Concat(std::move(a1), a2);

			// Verify strings were moved (source strings are empty, copies match merged values)
			CHECK(s1.empty());
			CHECK(s2.empty());
			CHECK(merged.data()[0] == s1_copy);
			CHECK(merged.data()[1] == s2_copy);

			// Verify string array was copied (source still has its values)
			CHECK(merged.data()[2] == "thirty");
			CHECK(merged.data()[3] == "forty");
			CHECK(a2.data()[0] == "thirty");
			CHECK(a2.data()[1] == "forty");
		}
	}

	SECTION("multiple arrays (3+)")
	{
		SECTION("Concat three int arrays")
		{
			KeyedArray<int, 1> a1{10};
			KeyedArray<int, 2, 3> a2{20, 30};
			KeyedArray<int, 4, 5, 6> a3{40, 50, 60};

			auto merged = Concat(a1, a2, a3);

			CHECK(merged.Size() == 6);
			CHECK(merged.Get<1>() == 10);
			CHECK(merged.Get<2>() == 20);
			CHECK(merged.Get<3>() == 30);
			CHECK(merged.Get<4>() == 40);
			CHECK(merged.Get<5>() == 50);
			CHECK(merged.Get<6>() == 60);
		}

		SECTION("Concat four arrays with struct keys")
		{
			KeyedArray<int, Point{0, 0}> a1{1};
			KeyedArray<int, Point{1, 0}> a2{2};
			KeyedArray<int, Point{0, 1}> a3{3};
			KeyedArray<int, Point{1, 1}> a4{4};

			auto merged = Concat(a1, a2, a3, a4);

			CHECK(merged.Size() == 4);
			CHECK(merged.Get<Point{0, 0}>() == 1);
			CHECK(merged.Get<Point{1, 0}>() == 2);
			CHECK(merged.Get<Point{0, 1}>() == 3);
			CHECK(merged.Get<Point{1, 1}>() == 4);
		}

		SECTION("Concat many single-element arrays")
		{
			KeyedArray<int, 1> a1{10};
			KeyedArray<int, 2> a2{20};
			KeyedArray<int, 3> a3{30};
			KeyedArray<int, 4> a4{40};
			KeyedArray<int, 5> a5{50};

			auto merged = Concat(a1, a2, a3, a4, a5);

			CHECK(merged.Size() == 5);
			CHECK(merged.Get<1>() == 10);
			CHECK(merged.Get<2>() == 20);
			CHECK(merged.Get<3>() == 30);
			CHECK(merged.Get<4>() == 40);
			CHECK(merged.Get<5>() == 50);
		}
	}

	SECTION("key order preservation")
	{
		SECTION("Key order follows declaration order of input arrays")
		{
			KeyedArray<int, 1, 2> a1{10, 20};
			KeyedArray<int, 3, 4> a2{30, 40};
			KeyedArray<int, 5, 6> a3{50, 60};

			auto merged = Concat(a1, a2, a3);

			// Verify keys are in order: a1's keys, then a2's keys, then a3's keys
			CHECK(merged.KeySet[0] == 1);
			CHECK(merged.KeySet[1] == 2);
			CHECK(merged.KeySet[2] == 3);
			CHECK(merged.KeySet[3] == 4);
			CHECK(merged.KeySet[4] == 5);
			CHECK(merged.KeySet[5] == 6);

			// Verify values match key positions
			CHECK(merged.data()[0] == 10);
			CHECK(merged.data()[1] == 20);
			CHECK(merged.data()[2] == 30);
			CHECK(merged.data()[3] == 40);
			CHECK(merged.data()[4] == 50);
			CHECK(merged.data()[5] == 60);
		}

		SECTION("Foreach visits merged keys in correct order")
		{
			KeyedArray<int, Point{1, 0}> a1{100};
			KeyedArray<int, Point{2, 0}, Point{0, 2}> a2{200, 300};

			auto merged = Concat(a1, a2);

			std::vector<Point> visited;
			merged.Foreach([&](const Point& k, const int&) { visited.push_back(k); });

			REQUIRE(visited.size() == 3);
			CHECK(visited[0] == Point{1, 0});
			CHECK(visited[1] == Point{2, 0});
			CHECK(visited[2] == Point{0, 2});
		}

		SECTION("Foreach<NTTP> visits merged keys in correct order")
		{
			KeyedArray<int, Point{1, 0}> a1{100};
			KeyedArray<int, Point{2, 0}, Point{0, 2}> a2{200, 300};

			{
				auto merged = Concat(a1, a2);

				std::vector<Point> visited;
				merged.Foreach([&]<Point K>(const int&) { visited.push_back(K); });

				REQUIRE(visited.size() == 3);
				CHECK(visited[0] == Point{1, 0});
				CHECK(visited[1] == Point{2, 0});
				CHECK(visited[2] == Point{0, 2});
			}
			{
				const auto merged = Concat(a1, a2);

				std::vector<Point> visited;
				merged.Foreach([&]<Point K>(const int&) { visited.push_back(K); });

				REQUIRE(visited.size() == 3);
				CHECK(visited[0] == Point{1, 0});
				CHECK(visited[1] == Point{2, 0});
				CHECK(visited[2] == Point{0, 2});
			}
		}
	}

	SECTION("constexpr")
	{
		SECTION("Concat is constexpr")
		{
			constexpr auto merge_constexpr = []() constexpr
			{
				KeyedArray<int, 1, 2> a1{10, 20};
				KeyedArray<int, 3, 4> a2{30, 40};
				return Concat(a1, a2);
			};

			constexpr auto merged = merge_constexpr();

			STATIC_CHECK(merged.Get<1>() == 10);
			STATIC_CHECK(merged.Get<2>() == 20);
			STATIC_CHECK(merged.Get<3>() == 30);
			STATIC_CHECK(merged.Get<4>() == 40);
			STATIC_CHECK(merged.Size() == 4);
		}
	}
#endif
}

#if __cplusplus >= 202002L
// a possible implementation of SecUtility::Merge
template <typename Op, typename Init, typename... KeyedArrays>
constexpr auto Merge_(Op op, const Init init, KeyedArrays&&... arrays)
    requires((Detail::KeyedArray::is_keyed_array<std::remove_cvref_t<KeyedArrays>>::value && ...))
{
	// if constexpr (requires { requires Concat(std::forward<KeyedArrays>(arrays)...); })
	// {
	// 	return Concat(std::forward<KeyedArrays>(arrays)...);
	// }
	// else
	{
		using Result = Detail::KeyedArray::merged_type<std::remove_cvref_t<KeyedArrays>...>::type;
		Result result{};

		for (std::size_t i = 0; i < Result::Size(); ++i)
		{
			result.data()[i] = init;
		}

		(arrays.Foreach(
		         [&](const auto& k, const auto& v)
		         {
			         const auto index = Detail::KeyedArray::IndexOf(Result::KeySet, k);
			         result.data()[index] = op(std::as_const(result.data()[index]), v);
		         }),
		 ...);

		return result;
	}
}
#endif

TEST_CASE("Merge(KeyedArray...)")
{
#if __cplusplus < 202002L
	SKIP("Test requires C++20");
#else
	SECTION("Plus")
	{
		constexpr auto merge_constexpr = []() constexpr
		{
			KeyedArray<int, 1, 2> a1{10, 20};
			KeyedArray<int, 4, 2> a2{42, 69};
			KeyedArray<int, 3, 4> a3{30, 40};
			// return Concat(a1, a2, a3);  // not compile
			return Merge_(std::plus<>{}, 0, a1, a2, a3);  // not compile
		};

		constexpr auto merged = merge_constexpr();

		STATIC_CHECK(merged.Get<1>() == 10);
		STATIC_CHECK(merged.Get<2>() == 20 + 69);
		STATIC_CHECK(merged.Get<3>() == 30);
		STATIC_CHECK(merged.Get<4>() == 42 + 40);
		STATIC_CHECK(merged.Size() == 4);
	}

	SECTION("Multiplies")
	{
		constexpr auto merge_constexpr = []() constexpr
		{
			KeyedArray<int, 1, 2> a1{10, 20};
			KeyedArray<int, 4, 2> a2{42, 69};
			KeyedArray<int, 3, 4> a3{30, 40};
			// return Concat(a1, a2, a3);  // not compile
			return Merge_(std::multiplies<>{}, 1, a1, a2, a3);  // not compile
		};

		constexpr auto merged = merge_constexpr();

		STATIC_CHECK(merged.Get<1>() == 10);
		STATIC_CHECK(merged.Get<2>() == 20 * 69);
		STATIC_CHECK(merged.Get<3>() == 30);
		STATIC_CHECK(merged.Get<4>() == 42 * 40);
		STATIC_CHECK(merged.Size() == 4);
	}
#endif
}
