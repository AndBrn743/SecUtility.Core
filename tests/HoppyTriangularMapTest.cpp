// SPDX-License-Identifier: MIT

// Triangular-compressed specification: D15; sections 5, 9-10, and 14-16.

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/TriangularCompressedMatrix.hpp>

#include <Eigen/Core>

#include <array>
#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace
{
	template <typename T, typename = void>
	struct has_resize : std::false_type
	{};
	template <typename T>
	struct has_resize<T, std::void_t<decltype(std::declval<T&>().resize(3))>> : std::true_type
	{};

	template <typename T, typename = void>
	struct has_coeff_ref : std::false_type
	{};
	template <typename T>
	struct has_coeff_ref<T, std::void_t<decltype(std::declval<T&>().coeffRef(0, 0))>> : std::true_type
	{};

	template <typename Map>
	struct is_supported_stride : std::false_type
	{};
	template <typename Plain, int Options>
	struct is_supported_stride<Eigen::Map<Plain, Options, Eigen::Stride<0, 0>>> : std::true_type
	{};

	using Plain = Hoppy::SymmetricMatrixXd;
	using MutableMap = Eigen::Map<Plain>;
	using ConstMap = Eigen::Map<const Plain>;
	using CustomStrideMap = Eigen::Map<Plain, Eigen::Unaligned,
	                                  Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
	static_assert(!has_resize<MutableMap>::value);
	static_assert(has_coeff_ref<MutableMap>::value);
	static_assert(!has_coeff_ref<ConstMap>::value);
	static_assert(is_supported_stride<MutableMap>::value);
	static_assert(!is_supported_stride<CustomStrideMap>::value);
	static_assert((Eigen::internal::traits<MutableMap>::Flags & Eigen::DirectAccessBit) == 0);
	static_assert((Eigen::internal::traits<ConstMap>::Flags & Eigen::LvalueBit) == 0);
	static_assert(std::is_constructible_v<Eigen::Map<Hoppy::SymmetricMatrix<double, 3>>, double*>);
	static_assert(std::is_constructible_v<Eigen::Map<const Hoppy::SymmetricMatrix<double, 3>>, const double*>);
	static_assert(!std::is_constructible_v<MutableMap, double*>);

	template <typename PlainMatrix>
	void requireMutableAndConstMap()
	{
		using Scalar = typename PlainMatrix::Scalar;
		alignas(16) std::array<Scalar, 6> buffer{};
		Eigen::Map<PlainMatrix> mutableMap(buffer.data(), 3);
		constexpr bool upperTriangular = std::is_same_v<typename PlainMatrix::StructureTag,
		                                               Hoppy::Detail::UpperTriangularTag>;
		const Eigen::Index row = upperTriangular ? 0 : 2;
		const Eigen::Index column = upperTriangular ? 2 : 0;
		mutableMap(row, column) = Scalar(7);
		Eigen::Map<const PlainMatrix> constMap(buffer.data(), 3);
		REQUIRE(constMap.rows() == 3);
		REQUIRE(constMap.storedSize() == 6);
		REQUIRE(constMap.coeff(row, column) == mutableMap.coeff(row, column));
	}
}  // namespace

TEST_CASE("mutable and const maps cover every family and packing side")
{
	using namespace Hoppy;
	using Complex = std::complex<double>;
	requireMutableAndConstMap<UpperTriangularMatrix<double>>();
	requireMutableAndConstMap<UpperTriangularMatrix<double, Eigen::Dynamic, TrianglePacking::Upper>>();
	requireMutableAndConstMap<LowerTriangularMatrix<double>>();
	requireMutableAndConstMap<LowerTriangularMatrix<double, Eigen::Dynamic, TrianglePacking::Upper>>();
	requireMutableAndConstMap<SymmetricMatrix<double>>();
	requireMutableAndConstMap<SymmetricMatrix<double, Eigen::Dynamic, TrianglePacking::Upper>>();
	requireMutableAndConstMap<AntiSymmetricMatrix<double>>();
	requireMutableAndConstMap<AntiSymmetricMatrix<double, Eigen::Dynamic, TrianglePacking::Upper>>();
	requireMutableAndConstMap<HermitianMatrix<Complex>>();
	requireMutableAndConstMap<HermitianMatrix<Complex, Eigen::Dynamic, TrianglePacking::Upper>>();
	requireMutableAndConstMap<AntiHermitianMatrix<Complex>>();
	requireMutableAndConstMap<AntiHermitianMatrix<Complex, Eigen::Dynamic, TrianglePacking::Upper>>();
}

TEST_CASE("map writes share reflected and constrained coefficient behavior")
{
	using Complex = std::complex<double>;
	alignas(16) Complex buffer[6]{};
	Eigen::Map<Hoppy::HermitianMatrix<Complex>> map(buffer, 3);
	map(0, 2) = Complex(3, 4);
	REQUIRE(buffer[3] == Complex(3, -4));
	REQUIRE(map.coeff(0, 2) == Complex(3, 4));
	map(0, 2) += Complex(1, 1);
	REQUIRE(map.coeff(0, 2) == Complex(4, 5));
}

TEST_CASE("fixed dynamic aligned and zero-sized map constructors are accepted")
{
	alignas(16) double buffer[6]{};
	Eigen::Map<Hoppy::SymmetricMatrix<double, 3>> fixed(buffer);
	Eigen::Map<Hoppy::SymmetricMatrixXd> dynamic(buffer, 3, 3);
	Eigen::Map<Hoppy::SymmetricMatrixXd, Eigen::Aligned> aligned(buffer, 3);
	Eigen::Map<Hoppy::SymmetricMatrixXd> empty(nullptr, 0);
	REQUIRE(fixed.dimension() == 3);
	REQUIRE(dynamic.dimension() == 3);
	REQUIRE(aligned.data() == buffer);
	REQUIRE(empty.storedSize() == 0);
	double scalarBuffer[1]{};
	Eigen::Map<Hoppy::SymmetricMatrix<double, 1>> scalar(scalarBuffer, 1, 1);
	scalar[0] = 9.0;
	const Eigen::Map<const Hoppy::SymmetricMatrix<double, 1>> constScalar(scalarBuffer);
	REQUIRE(constScalar(0) == 9.0);
}

TEST_CASE("owning and mapped structured copies support opposite packing and overlap")
{
	double buffer[] = {1, 2, 3, 4, 5, 6};
	Eigen::Map<Hoppy::SymmetricMatrixXd> lower(buffer, 3);
	using Upper = Hoppy::SymmetricMatrix<double, Eigen::Dynamic, Hoppy::TrianglePacking::Upper>;
	Upper owned(lower);
	REQUIRE(owned.coeff(0, 2) == 4.0);
	Eigen::Map<Upper> upper(buffer, 3);
	upper = lower;
	REQUIRE(upper.coeff(0, 2) == 4.0);
	Hoppy::SymmetricMatrixXd copied(upper);
	REQUIRE(copied.coeff(2, 1) == 5.0);
	auto* self = &lower;
	lower = *self;
	REQUIRE(lower.coeff(2, 1) == 5.0);
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("invalid map construction and writes assert")
{
	double buffer[6]{};
	REQUIRE_THROWS_AS(Eigen::Map<Hoppy::SymmetricMatrixXd>(nullptr, 3),
	                  Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(Eigen::Map<Hoppy::SymmetricMatrixXd>(buffer, 2, 3),
	                  Hoppy::Test::EigenAssertionFailure);
	using Fixed = Hoppy::SymmetricMatrix<double, 3>;
	REQUIRE_THROWS_AS(Eigen::Map<Fixed>(buffer, 2), Hoppy::Test::EigenAssertionFailure);

	alignas(16) std::byte raw[64]{};
	auto* insufficientlyAligned = reinterpret_cast<double*>(raw + alignof(double));
	REQUIRE_THROWS_AS((Eigen::Map<Hoppy::SymmetricMatrixXd, Eigen::Aligned>(
	                          insufficientlyAligned, 3)),
	                  Hoppy::Test::EigenAssertionFailure);

	Eigen::Map<Hoppy::UpperTriangularMatrixXd> upper(buffer, 3);
	REQUIRE_THROWS_AS(upper(2, 0) = 1.0, Hoppy::Test::EigenAssertionFailure);
	using Complex = std::complex<double>;
	Complex complexBuffer[6]{};
	Eigen::Map<Hoppy::AntiHermitianMatrix<Complex>> anti(complexBuffer, 3);
	REQUIRE_THROWS_AS(anti(1, 1) = Complex(1, 2), Hoppy::Test::EigenAssertionFailure);
}
#endif
