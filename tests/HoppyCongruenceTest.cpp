// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <complex>
#include <type_traits>
#include <utility>

namespace
{
	template <typename Matrix, typename Transform, typename = void>
	struct has_transformed_by : std::false_type
	{};

	template <typename Matrix, typename Transform>
	struct has_transformed_by<
	        Matrix, Transform,
	        std::void_t<decltype(std::declval<const Matrix&>().transformedBy(
	                std::declval<const Transform&>()))>> : std::true_type
	{};
}  // namespace

TEST_CASE("real congruence transforms use the specified operand order")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	matrix[0](0, 0) = 2.0;
	matrix[1] << 1.0, 2.0, -3.0, 4.0;
	Hoppy::BlockDiagonalMatrix<double> transform{1, 2};
	transform[0](0, 0) = -2.0;
	transform[1] << 1.0, 3.0, -2.0, 4.0;
	const Eigen::Matrix3d denseMatrix = matrix.toDense();
	const Eigen::Matrix3d denseTransform = transform.toDense();

	Hoppy::Test::requireApprox(matrix.transformedBy(transform).toDense(),
	                           denseTransform * denseMatrix * denseTransform.adjoint());
	Hoppy::Test::requireApprox(matrix.backTransformedBy(transform).toDense(),
	                           denseTransform.adjoint() * denseMatrix * denseTransform);
}

TEST_CASE("complex nonsymmetric transforms use adjoint rather than transpose")
{
	using Complex = std::complex<double>;
	Hoppy::BlockDiagonalMatrix<double> matrix{2};
	matrix[0] << 2.0, -1.0, 3.0, 4.0;
	Hoppy::BlockDiagonalMatrix<Complex> transform{2};
	transform[0] << Complex{1.0, 2.0}, Complex{-3.0, 1.0},
	                Complex{2.0, -4.0}, Complex{5.0, 3.0};
	const auto result = matrix.transformedBy(transform);
	static_assert(std::is_same_v<typename decltype(result)::Scalar, Complex>);
	Hoppy::Test::requireApprox(result.toDense(),
	                           transform.toDense() * matrix.toDense() * transform.toDense().adjoint());
	Hoppy::Test::requireApprox(matrix.backTransformedBy(transform).toDense(),
	                           transform.toDense().adjoint() * matrix.toDense() * transform.toDense());
}

TEST_CASE("lazy operands and nested congruence results own intermediate nodes")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	Hoppy::BlockDiagonalMatrix<double> transform{1, 2};
	matrix.setIdentity();
	transform.setConstant(0.5);
	const auto lazyTransform = transform + transform.transpose();
	const Eigen::Matrix3d denseLazyTransform = lazyTransform.toDense();
	const auto nested = matrix.transformedBy(lazyTransform).backTransformedBy(transform);
	const Eigen::Matrix3d first = denseLazyTransform * matrix.toDense()
	                              * denseLazyTransform.adjoint();

	Hoppy::Test::requireApprox(nested.toDense(),
	                           transform.toDense().adjoint() * first * transform.toDense());

	Hoppy::Test::requireApprox((-matrix.transformedBy(transform) / 2.0).toDense(),
	                           -(transform.toDense() * matrix.toDense()
	                             * transform.toDense().adjoint()) / 2.0);

	Hoppy::Test::requireApprox((matrix.transformedBy(transform) * matrix).toDense(),
	                           transform.toDense() * matrix.toDense()
	                                   * transform.toDense().adjoint() * matrix.toDense());

	REQUIRE(matrix.transformedBy(transform).trace()
	        == (transform.toDense() * matrix.toDense() * transform.toDense().adjoint()).trace());
}

TEST_CASE("in-place congruence is alias safe including self-congruence")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	matrix[0](0, 0) = 2.0;
	matrix[1] << 1.0, 2.0, 3.0, 5.0;
	Hoppy::BlockDiagonalMatrix<double> transform{1, 2};
	transform[0](0, 0) = 3.0;
	transform[1] << 2.0, -1.0, 4.0, 3.0;
	const Eigen::Matrix3d expected = transform.toDense() * matrix.toDense()
	                                 * transform.toDense().adjoint();
	matrix.transformBy(transform);
	Hoppy::Test::requireApprox(matrix.toDense(), expected);

	Hoppy::BlockDiagonalMatrix<double> self{1, 2};
	self[0](0, 0) = 2.0;
	self[1] << 1.0, 2.0, -1.0, 3.0;
	const Eigen::Matrix3d denseSelf = self.toDense();
	self.transformBy(self);
	Hoppy::Test::requireApprox(self.toDense(), denseSelf * denseSelf * denseSelf.adjoint());

	Hoppy::BlockDiagonalMatrix<double> shared{1, 2};
	shared.setIdentity();
	const Eigen::Matrix3d denseShared = shared.toDense();
	shared.backTransformBy(shared.transpose());
	Hoppy::Test::requireApprox(shared.toDense(),
	                           denseShared * denseShared * denseShared.transpose());
}

TEST_CASE("non-block-diagonal transforms are absent")
{
	using Matrix = Hoppy::BlockDiagonalMatrix<double>;
	static_assert(has_transformed_by<Matrix, Matrix>::value);
	static_assert(!has_transformed_by<Matrix, Eigen::MatrixXd>::value);
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("congruence transforms enforce identical blocking")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	Hoppy::BlockDiagonalMatrix<double> transform{3};
	REQUIRE_THROWS_AS((void)matrix.transformedBy(transform), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS((void)matrix.backTransformedBy(transform), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS((void)matrix.transformBy(transform), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS((void)matrix.backTransformBy(transform), Hoppy::Test::EigenAssertionFailure);
}

TEST_CASE("congruence evalTo detects overlap in either expression branch")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	Hoppy::BlockDiagonalMatrix<double> transform{1, 2};
	const auto expression = matrix.transformedBy(transform);
	Eigen::Map<Eigen::VectorXd> matrixStorage(matrix.data(), matrix.storedSize());
	Eigen::Map<Eigen::VectorXd> transformStorage(transform.data(), transform.storedSize());

	REQUIRE_THROWS_AS(expression.evalTo(matrixStorage), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(expression.evalTo(transformStorage), Hoppy::Test::EigenAssertionFailure);
}
#endif
