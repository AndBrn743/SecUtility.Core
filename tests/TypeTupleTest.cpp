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
