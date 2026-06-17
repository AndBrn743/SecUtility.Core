//
// Created by Andy on 6/17/2026.
//

#include <SecUtility/Collection/TypeTuple.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>


using namespace SecUtility;

#define DUAL_CHECK(...)                                                                                                \
	CHECK(__VA_ARGS__);                                                                                                \
	STATIC_CHECK(__VA_ARGS__)


TEST_CASE("TypePair is an alias for a two-element TypeTuple")
{
	STATIC_CHECK((std::is_same_v<TypePair<int, double>, TypeTuple<int, double>>));
	STATIC_CHECK((std::is_same_v<TypePair<void, void>, TypeTuple<void, void>>));
	STATIC_CHECK((std::is_same_v<TypePair<int, TypeTuple<double>>, TypeTuple<int, TypeTuple<double>>>));
}


TEST_CASE("ConcatenatedTypeTuple: identity for a single TypeTuple")
{
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<>>, TypeTuple<>>));
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<int>>, TypeTuple<int>>));
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<int, double, char>>, TypeTuple<int, double, char>>));
}


TEST_CASE("ConcatenatedTypeTuple: concatenates two TypeTuples")
{
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<>, TypeTuple<>>, TypeTuple<>>));
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<int>, TypeTuple<>>, TypeTuple<int>>));
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<>, TypeTuple<int>>, TypeTuple<int>>));
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<int>, TypeTuple<double>>, TypeTuple<int, double>>));
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<int, char>, TypeTuple<double, void>>,
	                             TypeTuple<int, char, double, void>>));
}


TEST_CASE("ConcatenatedTypeTuple: concatenates three or more TypeTuples")
{
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<int>, TypeTuple<double>, TypeTuple<char>>,
	                             TypeTuple<int, double, char>>));
	STATIC_CHECK(
	        (std::is_same_v<ConcatenatedTypeTuple<TypeTuple<int>, TypeTuple<double>, TypeTuple<char>, TypeTuple<void>>,
	                        TypeTuple<int, double, char, void>>));
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<int, int>, TypeTuple<double>, TypeTuple<char, char>>,
	                             TypeTuple<int, int, double, char, char>>));
}


TEST_CASE("ConcatenatedTypeTuple: preserves element order")
{
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<char, int, double>, TypeTuple<float>>,
	                             TypeTuple<char, int, double, float>>));
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<float>, TypeTuple<char, int, double>>,
	                             TypeTuple<float, char, int, double>>));
	STATIC_CHECK((std::is_same_v<
	              ConcatenatedTypeTuple<TypeTuple<short>, TypeTuple<int>, TypeTuple<long>, TypeTuple<long long>>,
	              TypeTuple<short, int, long, long long>>));
}


TEST_CASE("ConcatenatedTypeTuple: handles duplicate types without deduplication")
{
	STATIC_CHECK(
	        (std::is_same_v<ConcatenatedTypeTuple<TypeTuple<int, int>, TypeTuple<int>>, TypeTuple<int, int, int>>));
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<int, double>, TypeTuple<int, double>>,
	                             TypeTuple<int, double, int, double>>));
}


TEST_CASE("ConcatenatedTypeTuple: handles reference and pointer element types")
{
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<int&>, TypeTuple<const double&>>,
	                             TypeTuple<int&, const double&>>));
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<int*>, TypeTuple<const int*>>,
	                             TypeTuple<int*, const int*>>));
	STATIC_CHECK(
	        (std::is_same_v<ConcatenatedTypeTuple<TypeTuple<int&&>, TypeTuple<double&>>, TypeTuple<int&&, double&>>));
}


TEST_CASE("ConcatenatedTypeTuple: handles nested TypeTuple elements")
{
	STATIC_CHECK((std::is_same_v<ConcatenatedTypeTuple<TypeTuple<TypeTuple<int>>, TypeTuple<TypeTuple<double>>>,
	                             TypeTuple<TypeTuple<int>, TypeTuple<double>>>));
	STATIC_CHECK(
	        (std::is_same_v<ConcatenatedTypeTuple<TypeTuple<TypePair<int, double>>, TypeTuple<TypePair<char, float>>>,
	                        TypeTuple<TypePair<int, double>, TypePair<char, float>>>));
}


TEST_CASE("TypeTuple::TypeAt retrieves the element at a given index")
{
	STATIC_CHECK((std::is_same_v<TypeTuple<int>::TypeAt<0>, int>));
	STATIC_CHECK((std::is_same_v<TypeTuple<int, double>::TypeAt<0>, int>));
	STATIC_CHECK((std::is_same_v<TypeTuple<int, double>::TypeAt<1>, double>));
	STATIC_CHECK((std::is_same_v<TypeTuple<int, double, char>::TypeAt<0>, int>));
	STATIC_CHECK((std::is_same_v<TypeTuple<int, double, char>::TypeAt<1>, double>));
	STATIC_CHECK((std::is_same_v<TypeTuple<int, double, char>::TypeAt<2>, char>));
	STATIC_CHECK((std::is_same_v<TypeTuple<short, int, long, long long>::TypeAt<3>, long long>));
}


TEST_CASE("TypeTuple::TypeAt works through the TypePair alias")
{
	STATIC_CHECK((std::is_same_v<TypePair<int, double>::TypeAt<0>, int>));
	STATIC_CHECK((std::is_same_v<TypePair<int, double>::TypeAt<1>, double>));
}


TEST_CASE("TypeTuple::TypeAt handles void, reference, pointer, and rvalue-reference element types")
{
	STATIC_CHECK((std::is_same_v<TypeTuple<void>::TypeAt<0>, void>));
	STATIC_CHECK((std::is_same_v<TypeTuple<int&, const double&>::TypeAt<0>, int&>));
	STATIC_CHECK((std::is_same_v<TypeTuple<int&, const double&>::TypeAt<1>, const double&>));
	STATIC_CHECK((std::is_same_v<TypeTuple<int*, const int*>::TypeAt<1>, const int*>));
	STATIC_CHECK((std::is_same_v<TypeTuple<int&&, double&>::TypeAt<0>, int&&>));
}


TEST_CASE("TypeTuple::TypeAt handles nested TypeTuple and TypePair element types")
{
	STATIC_CHECK((std::is_same_v<TypeTuple<TypeTuple<int>, TypeTuple<double>>::TypeAt<0>, TypeTuple<int>>));
	STATIC_CHECK((std::is_same_v<TypeTuple<TypeTuple<int>, TypeTuple<double>>::TypeAt<1>, TypeTuple<double>>));
	STATIC_CHECK(
	        (std::is_same_v<TypeTuple<TypePair<int, double>, TypePair<char, float>>::TypeAt<1>, TypePair<char, float>>));
}
