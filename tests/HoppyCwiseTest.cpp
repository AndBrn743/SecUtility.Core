// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <complex>
#include <limits>
#include <type_traits>
#include <utility>

namespace
{
	template <typename Lhs, typename Rhs, typename = void>
	struct has_addition : std::false_type {};
	template <typename Lhs, typename Rhs>
	struct has_addition<Lhs, Rhs, std::void_t<decltype(std::declval<const Lhs&>() + std::declval<const Rhs&>())>>
	    : std::true_type {};

	template <typename Lhs, typename Rhs, typename = void>
	struct has_add_assignment : std::false_type {};
	template <typename Lhs, typename Rhs>
	struct has_add_assignment<Lhs, Rhs,
	                          std::void_t<decltype(std::declval<Lhs&>() += std::declval<const Rhs&>())>>
	    : std::true_type {};
	template <typename Expression, typename Scalar, typename = void>
	struct has_right_multiply : std::false_type {};
	template <typename Expression, typename Scalar>
	struct has_right_multiply<Expression, Scalar,
	                          std::void_t<decltype(std::declval<const Expression&>()
	                                             * std::declval<const Scalar&>())>> : std::true_type {};

	struct UnsupportedScalar {};

	template <typename Expression>
	using expression_scalar = typename Eigen::internal::traits<Expression>::Scalar;

	template <typename T>
	void fill(T& value)
	{
		for (Eigen::Index index = 0; index < value.storedSize(); ++index)
			value.data()[index] = static_cast<typename T::Scalar>(index + 1);
	}
}

TEST_CASE("like-kind cwise expressions are lazy and match dense Eigen")
{
	Hoppy::BlockDiagonalMatrix<double> lhs{1, 2, 3};
	Hoppy::BlockDiagonalMatrix<double> rhs{1, 2, 3};
	fill(lhs);
	fill(rhs);
	rhs *= 2.0;
	const auto lhsDense = lhs.toDense();
	const auto rhsDense = rhs.toDense();

	const auto sum = lhs + rhs;
	const auto difference = lhs - rhs;
	Hoppy::Test::requireApprox(sum.toDense(), lhsDense + rhsDense);
	Hoppy::Test::requireApprox(difference.toDense(), lhsDense - rhsDense);
	Hoppy::Test::requireApprox((lhs * 2.5).toDense(), lhsDense * 2.5);
	Hoppy::Test::requireApprox((2.5 * lhs).toDense(), 2.5 * lhsDense);
	Hoppy::Test::requireApprox((lhs / 2.0).toDense(), lhsDense / 2.0);

	lhs[2](0, 0) = 99.0;
	REQUIRE(sum[0](0, 0) == lhs[0](0, 0) + rhs[0](0, 0));
	REQUIRE(sum[2](0, 0) == 99.0 + rhs[2](0, 0));
}

TEST_CASE("vector cwise algebra preserves orientation and supports nested lifetime")
{
	Hoppy::BlockVector<double> lhs{1, 2, 3};
	Hoppy::BlockVector<double> rhs{1, 2, 3};
	fill(lhs);
	fill(rhs);
	const auto expected = (lhs.asDense() + rhs.asDense()).eval();

	Hoppy::BlockVector<double> nested = (lhs + rhs) * 3.0 - rhs;
	Hoppy::Test::requireApprox(nested.asDense(), expected * 3.0 - rhs.asDense());

	lhs = lhs + lhs;
	Hoppy::Test::requireApprox(lhs.asDense(), expected);
	lhs += lhs;
	Hoppy::Test::requireApprox(lhs.asDense(), expected * 2.0);
	lhs -= lhs;
	REQUIRE(lhs.asDense().isZero());

	Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Row> row{1, 2, 3};
	fill(row);
	using RowSum = decltype(row + row);
	static_assert(std::is_same_v<typename Eigen::internal::traits<RowSum>::Orientation, Hoppy::BlockVectorOrientation::Row>);
	Hoppy::Test::requireApprox((row + row).toDense(), row.asDense() + row.asDense());
}

TEST_CASE("result scalars follow the underlying Eigen operations")
{
	using RealVector = Hoppy::BlockVector<double>;
	using ComplexVector = Hoppy::BlockVector<std::complex<double>>;
	using FloatMatrix = Hoppy::BlockDiagonalMatrix<float>;

	using MixedSum = decltype(std::declval<const RealVector&>() + std::declval<const ComplexVector&>());
	using DenseMixedSum = decltype(std::declval<const Eigen::VectorXd&>()
	                             + std::declval<const Eigen::VectorXcd&>());
	static_assert(std::is_same_v<expression_scalar<MixedSum>, typename DenseMixedSum::Scalar>);

	using MixedDifference = decltype(std::declval<const FloatMatrix&>() - std::declval<const FloatMatrix&>());
	using DenseMixedDifference = decltype(std::declval<const Eigen::MatrixXf&>()
	                                    - std::declval<const Eigen::MatrixXf&>());
	static_assert(std::is_same_v<expression_scalar<MixedDifference>, typename DenseMixedDifference::Scalar>);

	using RightProduct = decltype(std::declval<const RealVector&>() * std::declval<double>());
	using LeftProduct = decltype(std::declval<double>() * std::declval<const RealVector&>());
	using Quotient = decltype(std::declval<const RealVector&>() / std::declval<double>());
	static_assert(std::is_same_v<expression_scalar<RightProduct>,
	                             typename decltype(std::declval<const Eigen::VectorXd&>() * double{})::Scalar>);
	static_assert(std::is_same_v<expression_scalar<LeftProduct>,
	                             typename decltype(double{} * std::declval<const Eigen::VectorXd&>())::Scalar>);
	static_assert(std::is_same_v<expression_scalar<Quotient>,
	                             typename decltype(std::declval<const Eigen::VectorXd&>() / double{})::Scalar>);
	using ProductOp = Eigen::internal::scalar_product_op<double, double>;
	using QuotientOp = Eigen::internal::scalar_quotient_op<double, double>;
	static_assert(std::is_same_v<expression_scalar<RightProduct>,
	                             typename Eigen::ScalarBinaryOpTraits<double, double, ProductOp>::ReturnType>);
	static_assert(std::is_same_v<expression_scalar<Quotient>,
	                             typename Eigen::ScalarBinaryOpTraits<double, double, QuotientOp>::ReturnType>);
}

TEST_CASE("integer scalar algebra follows Eigen coefficient semantics")
{
	Hoppy::BlockDiagonalMatrix<int> matrix{1, 2};
	fill(matrix);
	const auto dense = matrix.toDense();
	Hoppy::Test::requireApprox((matrix * 3).toDense(), dense * 3);
	Hoppy::Test::requireApprox((matrix / 2).toDense(), dense / 2);
}

TEST_CASE("unsupported kind and orientation combinations are absent")
{
	using Matrix = Hoppy::BlockDiagonalMatrix<double>;
	using ColumnVector = Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Column>;
	using RowVector = Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Row>;
	static_assert(!has_addition<Matrix, ColumnVector>::value);
	static_assert(!has_addition<ColumnVector, Matrix>::value);
	static_assert(!has_addition<ColumnVector, RowVector>::value);
	static_assert(!has_add_assignment<ColumnVector, RowVector>::value);
	static_assert(!has_addition<Hoppy::BlockVector<float>, Hoppy::BlockVector<double>>::value);
	static_assert(!has_right_multiply<ColumnVector, UnsupportedScalar>::value);
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("cwise and compound operations reject mismatched blocking")
{
	Hoppy::BlockDiagonalMatrix<double> matrixA{1, 2};
	Hoppy::BlockDiagonalMatrix<double> matrixB{3};
	REQUIRE_THROWS_AS((void)(matrixA + matrixB), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(matrixA += matrixB, Hoppy::Test::EigenAssertionFailure);

	Hoppy::BlockVector<double> vectorA{1, 2};
	Hoppy::BlockVector<double> vectorB{3};
	REQUIRE_THROWS_AS((void)(vectorA - vectorB), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(vectorA -= vectorB, Hoppy::Test::EigenAssertionFailure);
}
#endif

TEST_CASE("floating zero division follows Eigen without a Hoppy assertion")
{
	Hoppy::BlockVector<double> vector{1};
	vector[0](0) = 1.0;
	vector /= 0.0;
	REQUIRE(vector[0](0) == std::numeric_limits<double>::infinity());
}
