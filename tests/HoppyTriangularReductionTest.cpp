// SPDX-License-Identifier: MIT

// Triangular-compressed specification: sections 7.1, 9, 12.2, and 16.

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/TriangularCompressedMatrix.hpp>

#include <Eigen/Core>

#include <complex>
#include <limits>
#include <type_traits>
#include <utility>

namespace
{
	template <typename T, typename = void> struct has_max_coeff : std::false_type {};
	template <typename T>
	struct has_max_coeff<T, std::void_t<decltype(std::declval<const T&>().maxCoeff())>> : std::true_type {};
	static_assert(has_max_coeff<Hoppy::SymmetricMatrixXd>::value);
	static_assert(!has_max_coeff<Hoppy::HermitianMatrixXcd>::value);
}

TEST_CASE("reductions observe the represented dense matrix")
{
	Hoppy::SymmetricMatrix<double, 3> matrix;
	matrix.setZero();
	matrix(0, 0) = -2;
	matrix(1, 1) = 4;
	matrix(2, 2) = 6;
	matrix(0, 1) = 3;
	matrix(0, 2) = -5;
	matrix(1, 2) = 2;
	const auto dense = matrix.toDense();
	REQUIRE(matrix.sum() == dense.sum());
	REQUIRE(matrix.trace() == dense.trace());
	REQUIRE(matrix.squaredNorm() == dense.squaredNorm());
	REQUIRE(matrix.norm() == dense.norm());
	REQUIRE(matrix.mean() == dense.mean());
	REQUIRE(matrix.maxCoeff() == dense.maxCoeff());
	REQUIRE(matrix.minCoeff() == dense.minCoeff());
	REQUIRE(matrix.maxAbsCoeff() == 6);
	REQUIRE(matrix.allFinite());
	REQUIRE_FALSE(matrix.hasNaN());
	REQUIRE(matrix.isApprox(dense));
	REQUIRE(matrix.transpose().isApprox(dense.transpose()));
}

TEST_CASE("anti-family reflected terms and complex norms are counted logically")
{
	using Complex = std::complex<double>;
	Hoppy::AntiHermitianMatrix<Complex, 3> matrix;
	matrix.setZero();
	matrix(0, 0) = Complex(0, 2);
	matrix(0, 1) = Complex(3, 4);
	matrix(1, 2) = Complex(-1, 2);
	const auto dense = matrix.toDense();
	REQUIRE(matrix.sum() == dense.sum());
	REQUIRE(matrix.trace() == dense.trace());
	REQUIRE(matrix.squaredNorm() == dense.squaredNorm());
	REQUIRE(matrix.maxAbsCoeff() == 5);
	REQUIRE(matrix.adjoint().squaredNorm() == dense.adjoint().squaredNorm());
}

TEST_CASE("empty reductions have their specified identities")
{
	Hoppy::SymmetricMatrixXd empty;
	REQUIRE(empty.sum() == 0);
	REQUIRE(empty.trace() == 0);
	REQUIRE(empty.squaredNorm() == 0);
	REQUIRE(empty.norm() == 0);
	REQUIRE(empty.allFinite());
	REQUIRE_FALSE(empty.hasNaN());
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("undefined empty reductions assert")
{
	Hoppy::SymmetricMatrixXd empty;
	REQUIRE_THROWS_AS(empty.mean(), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(empty.maxCoeff(), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(empty.minCoeff(), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(empty.maxAbsCoeff(), Hoppy::Test::EigenAssertionFailure);
	Hoppy::SymmetricMatrixXd sized(2);
	REQUIRE_THROWS_AS(sized.isApprox(empty), Hoppy::Test::EigenAssertionFailure);
}
#else
TEST_CASE("no-debug mismatched approximation returns false")
{
	Hoppy::SymmetricMatrixXd empty;
	Hoppy::SymmetricMatrixXd sized(2);
	REQUIRE_FALSE(sized.isApprox(empty));
}
#endif

TEST_CASE("NaN infinity maps and packing variants participate in predicates")
{
	auto matrix = Hoppy::UpperTriangularMatrix<double, Eigen::Dynamic, Hoppy::TrianglePacking::Upper>::Zero(2);
	matrix(0, 1) = (std::numeric_limits<double>::infinity)();
	REQUIRE_FALSE(matrix.allFinite());
	matrix(0, 1) = (std::numeric_limits<double>::quiet_NaN)();
	REQUIRE(matrix.hasNaN());

	double storage[3] = {1, 2, 3};
	Eigen::Map<const Hoppy::SymmetricMatrix<double, 2>> map(storage);
	REQUIRE(map.sum() == map.toDense().sum());
}
