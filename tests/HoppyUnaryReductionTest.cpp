// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <complex>
#include <type_traits>

namespace
{
	template <typename T, typename = void>
	struct has_max_coeff : std::false_type {};
	template <typename T>
	struct has_max_coeff<T, std::void_t<decltype(std::declval<const T&>().maxCoeff())>> : std::true_type {};
}

TEST_CASE("lazy unary BD operations preserve blocking and match dense Eigen")
{
	using Complex = std::complex<double>;
	Hoppy::BlockDiagonalMatrix<Complex> matrix{1, 2};
	matrix[0](0, 0) = {1.0, 2.0};
	matrix[1] << Complex{2, -1}, Complex{3, 4}, Complex{5, -2}, Complex{-1, 3};
	const auto dense = matrix.toDense();
	Hoppy::Test::requireApprox((-matrix).toDense(), -dense);
	Hoppy::Test::requireApprox(matrix.transpose().toDense(), dense.transpose());
	Hoppy::Test::requireApprox(matrix.conjugate().toDense(), dense.conjugate());
	Hoppy::Test::requireApprox(matrix.adjoint().toDense(), dense.adjoint());
	Hoppy::Test::requireApprox(matrix.real().toDense(), dense.real());
	Hoppy::Test::requireApprox(matrix.imag().toDense(), dense.imag());
	Hoppy::Test::requireApprox(matrix.cast<std::complex<float>>().toDense(), dense.cast<std::complex<float>>());
	Hoppy::Test::requireApprox(matrix.transpose().conjugate().toDense(), dense.transpose().conjugate());
	static_assert(std::is_same_v<typename decltype(matrix.real())::Scalar, double>);
}

TEST_CASE("BV transpose and adjoint flip orientation while conjugate preserves it")
{
	using Complex = std::complex<double>;
	Hoppy::BlockVector<Complex> vector{1, 2};
	vector.asDense() << Complex{1, 2}, Complex{3, 4}, Complex{5, 6};
	const auto transposed = vector.transpose();
	const auto adjoint = vector.adjoint();
	const auto conjugated = vector.conjugate();
	static_assert(std::is_same_v<typename Eigen::internal::traits<decltype(transposed)>::Orientation, Hoppy::Row>);
	static_assert(std::is_same_v<typename Eigen::internal::traits<decltype(adjoint)>::Orientation, Hoppy::Row>);
	static_assert(std::is_same_v<typename Eigen::internal::traits<decltype(conjugated)>::Orientation, Hoppy::Column>);
	Hoppy::Test::requireApprox(transposed.toDense(), vector.asDense().transpose());
	Hoppy::Test::requireApprox(adjoint.toDense(), vector.asDense().adjoint());
	Hoppy::Test::requireApprox(conjugated.toDense(), vector.asDense().conjugate());
}

TEST_CASE("reductions match represented dense values and define empty identities")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	matrix[0](0, 0) = -2.0;
	matrix[1] << 1.0, 2.0, 3.0, 4.0;
	const auto dense = matrix.toDense();
	REQUIRE(matrix.sum() == dense.sum());
	REQUIRE(matrix.squaredNorm() == dense.squaredNorm());
	REQUIRE(matrix.norm() == dense.norm());
	REQUIRE(matrix.rootMeanSquare() == std::sqrt(dense.squaredNorm() / dense.size()));
	REQUIRE(matrix.mean() == dense.mean());
	REQUIRE(matrix.maxCoeff() == dense.maxCoeff());
	REQUIRE(matrix.minCoeff() == dense.minCoeff());
	REQUIRE(matrix.maxAbsCoeff() == dense.cwiseAbs().maxCoeff());
	REQUIRE(matrix.trace() == dense.trace());
	REQUIRE(matrix.determinant() == matrix[0].determinant() * matrix[1].determinant());
	REQUIRE(matrix.allFinite());
	REQUIRE_FALSE(matrix.hasNaN());
	REQUIRE(matrix.isApprox(matrix));

	const Hoppy::BlockDiagonalMatrix<double> empty;
	REQUIRE(empty.sum() == 0.0);
	REQUIRE(empty.trace() == 0.0);
	REQUIRE(empty.determinant() == 1.0);
	REQUIRE(empty.rootMeanSquare() == 0.0);
	REQUIRE(empty.allFinite());
	REQUIRE_FALSE(empty.hasNaN());
	static_assert(!has_max_coeff<Hoppy::BlockDiagonalMatrix<std::complex<double>>>::value);
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("empty ordered reductions assert")
{
	const Hoppy::BlockVector<double> empty;
	REQUIRE_THROWS_AS(empty.mean(), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(empty.maxCoeff(), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(empty.minCoeff(), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(empty.maxAbsCoeff(), Hoppy::Test::EigenAssertionFailure);
}
#endif
