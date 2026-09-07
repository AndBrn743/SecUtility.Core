// SPDX-License-Identifier: MIT

// Triangular-compressed specification: sections 7.1, 9, 12.2, and 16.

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/TriangularCompressedMatrix.hpp>

#include <Eigen/Core>

#include <complex>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
	template <typename T, typename = void> struct has_max_coeff : std::false_type {};
	template <typename T>
	struct has_max_coeff<T, std::void_t<decltype(std::declval<const T&>().maxCoeff())>> : std::true_type {};
	static_assert(has_max_coeff<Hoppy::SymmetricMatrixXd>::value);
	static_assert(!has_max_coeff<Hoppy::HermitianMatrixXcd>::value);

	template <Hoppy::TrianglePacking Packing>
	struct CountingExpression : Hoppy::TriangularCompressedMatrixExpr<CountingExpression<Packing>>
	{
		using Scalar = double;
		using StructureTag = Hoppy::Detail::SymmetricTag;
		static constexpr Hoppy::TrianglePacking PackingValue = Packing;
		Eigen::Index dimension() const { return 3; }
		Eigen::Index rows() const { return 3; }
		Eigen::Index cols() const { return 3; }
		Eigen::Index size() const { return 9; }
		double coeff(Eigen::Index row, Eigen::Index column) const
		{
			visits.emplace_back(row, column);
			return static_cast<double>(10 * row + column + 1);
		}
		mutable std::vector<std::pair<Eigen::Index, Eigen::Index>> visits;
	};
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

	Hoppy::AntiSymmetricMatrix<double, 2> m2;
	m2(0, 1) = -10;
	REQUIRE(m2.maxCoeff() == 10);
}

TEST_CASE("reductions visit independent coefficients once in packed order")
{
	CountingExpression<Hoppy::TrianglePacking::Lower> lower;
	(void) lower.squaredNorm();
	const std::vector<std::pair<Eigen::Index, Eigen::Index>> expectedLower{
	        {0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {2, 2}};
	REQUIRE(lower.visits == expectedLower);

	CountingExpression<Hoppy::TrianglePacking::Upper> upper;
	(void) upper.sum();
	const std::vector<std::pair<Eigen::Index, Eigen::Index>> expectedUpper{
	        {0, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}, {2, 2}};
	REQUIRE(upper.visits == expectedUpper);
}

TEST_CASE("lazy transform expressions exercise the generic reduction implementation")
{
	Hoppy::SymmetricMatrix<double, 3> matrix;
	matrix.setZero();
	matrix(0, 0) = -2;
	matrix(1, 1) = 4;
	matrix(2, 2) = 6;
	matrix(0, 1) = 3;
	matrix(0, 2) = -5;
	matrix(1, 2) = 2;
	const auto expression = matrix.transpose();
	const auto dense = matrix.toDense().transpose().eval();
	REQUIRE(expression.sum() == dense.sum());
	REQUIRE(expression.trace() == dense.trace());
	REQUIRE(expression.squaredNorm() == dense.squaredNorm());
	REQUIRE(expression.norm() == dense.norm());
	REQUIRE(expression.mean() == dense.mean());
	REQUIRE(expression.maxCoeff() == dense.maxCoeff());
	REQUIRE(expression.minCoeff() == dense.minCoeff());
	REQUIRE(expression.maxAbsCoeff() == 6);
	REQUIRE(expression.allFinite());
	REQUIRE_FALSE(expression.hasNaN());
	REQUIRE(expression.isApprox(dense));
	REQUIRE(expression.isApprox(dense, 0.0));

	Eigen::MatrixXd different = dense;
	different(0, 0) += 1;
	REQUIRE_FALSE(expression.isApprox(different, 0.0));
}

TEST_CASE("generic expression predicates observe nonfinite independent coefficients")
{
	auto matrix = Hoppy::UpperTriangularMatrix<double, 2>::Zero();
	matrix(0, 1) = (std::numeric_limits<double>::infinity)();
	const auto infinite = matrix.transpose();
	REQUIRE_FALSE(infinite.allFinite());
	REQUIRE_FALSE(infinite.hasNaN());
	matrix(0, 1) = (std::numeric_limits<double>::quiet_NaN)();
	const auto nan = matrix.adjoint();
	REQUIRE_FALSE(nan.allFinite());
	REQUIRE(nan.hasNaN());
}

TEST_CASE("empty lazy expressions retain reduction identities")
{
	Hoppy::SymmetricMatrixXd matrix;
	const auto expression = matrix.transpose();
	REQUIRE(expression.sum() == 0);
	REQUIRE(expression.trace() == 0);
	REQUIRE(expression.squaredNorm() == 0);
	REQUIRE(expression.norm() == 0);
	REQUIRE(expression.allFinite());
	REQUIRE_FALSE(expression.hasNaN());
}

TEST_CASE("triangular reductions include structural zeros without visiting them")
{
	auto matrix = Hoppy::UpperTriangularMatrix<double, 3, Hoppy::TrianglePacking::Lower>::Constant(-4);
	REQUIRE(matrix.sum() == -24);
	REQUIRE(matrix.maxCoeff() == 0);
	REQUIRE(matrix.minCoeff() == -4);
	REQUIRE(matrix.squaredNorm() == 96);
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

TEST_CASE("undefined empty lazy-expression reductions assert")
{
	Hoppy::SymmetricMatrixXd matrix;
	const auto expression = matrix.adjoint();
	REQUIRE_THROWS_AS(expression.mean(), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(expression.maxCoeff(), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(expression.minCoeff(), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(expression.maxAbsCoeff(), Hoppy::Test::EigenAssertionFailure);
	Hoppy::SymmetricMatrixXd sized(2);
	REQUIRE_THROWS_AS(expression.isApprox(sized.transpose()),
	                  Hoppy::Test::EigenAssertionFailure);
}
#else
TEST_CASE("no-debug mismatched approximation returns false")
{
	Hoppy::SymmetricMatrixXd empty;
	Hoppy::SymmetricMatrixXd sized(2);
	REQUIRE_FALSE(sized.isApprox(empty));
	REQUIRE_FALSE(sized.transpose().isApprox(empty.transpose()));
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
