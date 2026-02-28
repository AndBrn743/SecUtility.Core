//
// Created by Andy on 2/28/2026.
//

#include <SecUtility/Meta/ParameterPackUtility.hpp>
#include <catch2/catch_test_macros.hpp>


using namespace SecUtility;

#define DUAL_CHECK(...) CHECK(__VA_ARGS__); STATIC_CHECK(__VA_ARGS__)
#define DUAL_CHECK_FALSE(...) CHECK_FALSE(__VA_ARGS__); STATIC_CHECK_FALSE(__VA_ARGS__)

TEST_CASE("ParameterPackUtility::Get<I>(args...)")
{
	DUAL_CHECK([](auto... args) { return ParameterPackUtility::Get<0>(args...); }(42, 69, 73) == 42);
	DUAL_CHECK([](auto... args) { return ParameterPackUtility::Get<1>(args...); }(42, 69, 73) == 69);
	DUAL_CHECK([](auto... args) { return ParameterPackUtility::Get<2>(args...); }(42, 69, 73) == 73);
}

TEST_CASE("TypeAt<I, Ts...>")
{
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::TypeAt<0, void>, void>);
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::TypeAt<0, int>, int>);

	DUAL_CHECK(std::is_same_v<ParameterPackUtility::TypeAt<0, void, int, void, double, char>, void>);
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::TypeAt<1, void, int, void, double, char>, int>);
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::TypeAt<2, void, int, void, double, char>, void>);
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::TypeAt<3, void, int, void, double, char>, double>);
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::TypeAt<4, void, int, void, double, char>, char>);
}

TEST_CASE("InstanceCountIn<T, Ts...>")
{
	DUAL_CHECK(ParameterPackUtility::InstanceCountIn<void, void> == 1);
	DUAL_CHECK(ParameterPackUtility::InstanceCountIn<int, int> == 1);
	DUAL_CHECK(ParameterPackUtility::InstanceCountIn<int, int, int, int> == 3);

	DUAL_CHECK(ParameterPackUtility::InstanceCountIn<void, void, int, void, double, char> == 2);
	DUAL_CHECK(ParameterPackUtility::InstanceCountIn<int, void, int, void, double, char> == 1);
	DUAL_CHECK(ParameterPackUtility::InstanceCountIn<double, void, int, void, double, char> == 1);
	DUAL_CHECK(ParameterPackUtility::InstanceCountIn<char, void, int, void, double, char> == 1);
}


TEST_CASE("InstanceIndicesIn<T, Ts...>")
{
	// apparently operator== between two std::array's is not constexpr
	DUAL_CHECK(ParameterPackUtility::InstanceIndicesIn<void, void>[0] == 0);
	DUAL_CHECK(ParameterPackUtility::InstanceIndicesIn<int, int>[0] == 0);
	DUAL_CHECK(ParameterPackUtility::InstanceIndicesIn<int, int, int, int>[0] == 0);
	DUAL_CHECK(ParameterPackUtility::InstanceIndicesIn<int, int, int, int>[1] == 1);
	DUAL_CHECK(ParameterPackUtility::InstanceIndicesIn<int, int, int, int>[2] == 2);

	DUAL_CHECK(ParameterPackUtility::InstanceIndicesIn<void, void, int, void, double, char, std::void_t<>, int, int>[0] == 0);
	DUAL_CHECK(ParameterPackUtility::InstanceIndicesIn<void, void, int, void, double, char, std::void_t<>, int, int>[1] == 2);
	DUAL_CHECK(ParameterPackUtility::InstanceIndicesIn<void, void, int, void, double, char, std::void_t<>, int, int>[2] == 5);

	DUAL_CHECK(ParameterPackUtility::InstanceIndicesIn<int, void, int, void, double, char, std::void_t<>, int, int>[0] == 1);
	DUAL_CHECK(ParameterPackUtility::InstanceIndicesIn<int, void, int, void, double, char, std::void_t<>, int, int>[1] == 6);
	DUAL_CHECK(ParameterPackUtility::InstanceIndicesIn<int, void, int, void, double, char, std::void_t<>, int, int>[2] == 7);

	DUAL_CHECK(ParameterPackUtility::InstanceIndicesIn<double, void, int, void, double, char, std::void_t<>, int, int>[0] == 3);
	DUAL_CHECK(ParameterPackUtility::InstanceIndicesIn<char, void, int, void, double, char, std::void_t<>, int, int>[0] == 4);
}

TEST_CASE("HasDuplicate<Ts...>")
{
	DUAL_CHECK_FALSE(ParameterPackUtility::HasDuplicate<>);
	DUAL_CHECK_FALSE(ParameterPackUtility::HasDuplicate<int>);
	DUAL_CHECK_FALSE(ParameterPackUtility::HasDuplicate<void>);
	DUAL_CHECK_FALSE(ParameterPackUtility::HasDuplicate<void, int>);
	DUAL_CHECK_FALSE(ParameterPackUtility::HasDuplicate<int, void>);

	DUAL_CHECK(ParameterPackUtility::HasDuplicate<void, void>);
	DUAL_CHECK(ParameterPackUtility::HasDuplicate<int, int>);
	DUAL_CHECK(ParameterPackUtility::HasDuplicate<int, void, int>);
}

TEST_CASE("UniqueTypeTuple")
{
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::UniqueTypeTuple<>, std::tuple<>>);
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::UniqueTypeTuple<void>, std::tuple<void>>);

	DUAL_CHECK(std::is_same_v<ParameterPackUtility::UniqueTypeTuple<int>, std::tuple<int>>);
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::UniqueTypeTuple<int, int>, std::tuple<int>>);
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::UniqueTypeTuple<int, int, int>, std::tuple<int>>);
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::UniqueTypeTuple<int, int, int, int>, std::tuple<int>>);

	DUAL_CHECK(std::is_same_v<ParameterPackUtility::UniqueTypeTuple<double, double>, std::tuple<double>>);
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::UniqueTypeTuple<double, double, double>, std::tuple<double>>);

	DUAL_CHECK(std::is_same_v<ParameterPackUtility::UniqueTypeTuple<int, double>, std::tuple<int, double>>);
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::UniqueTypeTuple<double, int, double>, std::tuple<int, double>>);
	DUAL_CHECK(std::is_same_v<ParameterPackUtility::UniqueTypeTuple<int, char, int, double>, std::tuple<char, int, double>>);
}
