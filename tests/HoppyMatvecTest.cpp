// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <complex>
#include <type_traits>
#include <utility>

namespace
{
	template <typename Lhs, typename Rhs, typename = void>
	struct has_product : std::false_type {};
	template <typename Lhs, typename Rhs>
	struct has_product<Lhs, Rhs,
	                   std::void_t<decltype(std::declval<const Lhs&>() * std::declval<const Rhs&>())>>
	    : std::true_type {};

	template <typename Matrix>
	void fill(Matrix& matrix)
	{
		for (Eigen::Index index = 0; index < matrix.storedSize(); ++index)
			matrix.data()[index] = static_cast<typename Matrix::Scalar>(index + 1);
	}
}

TEST_CASE("block matrix times column block vector is eager and oriented")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2, 3};
	Hoppy::BlockVector<double> vector{1, 2, 3};
	fill(matrix);
	vector.asDense() << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
	const auto result = matrix * vector;
	const auto lazyOperands = (matrix + matrix) * (vector + vector);

	static_assert(std::is_same_v<std::remove_cv_t<decltype(result)>, Hoppy::BlockVector<double>>);
	static_assert(std::is_same_v<typename decltype(result)::Orientation, Hoppy::Column>);
	REQUIRE(result.blockingInfo() == matrix.blockingInfo());
	Hoppy::Test::requireApprox(result.asDense(), matrix.toDense() * vector.asDense());
	Hoppy::Test::requireApprox(lazyOperands.asDense(),
	                           (matrix.toDense() + matrix.toDense()) * (vector.asDense() + vector.asDense()));
}

TEST_CASE("row block vector times block matrix is eager and oriented")
{
	using Complex = std::complex<double>;
	Hoppy::BlockDiagonalMatrix<Complex> matrix{1, 2};
	Hoppy::BlockVector<double, Hoppy::Row> vector{1, 2};
	fill(matrix);
	vector.asDense() << 2.0, 3.0, 4.0;
	const auto result = vector * matrix;

	using DenseResult = decltype(vector.asDense() * matrix.toDense());
	static_assert(std::is_same_v<typename decltype(result)::Scalar, typename DenseResult::Scalar>);
	static_assert(std::is_same_v<typename decltype(result)::Orientation, Hoppy::Row>);
	Hoppy::Test::requireApprox(result.asDense(), vector.asDense() * matrix.toDense());
}

TEST_CASE("fixed-shape dense column and row expressions select eager paths")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	fill(matrix);
	Eigen::Vector3d column;
	column << 1.0, 2.0, 3.0;
	Eigen::RowVector3d row = column.transpose();

	const auto columnResult = matrix * (column + column);
	const auto rowResult = row.segment<3>(0) * matrix;
	static_assert(std::is_same_v<std::remove_cv_t<decltype(columnResult)>, Hoppy::BlockVector<double>>);
	static_assert(std::is_same_v<std::remove_cv_t<decltype(rowResult)>,
	                             Hoppy::BlockVector<double, Hoppy::Row>>);
	Hoppy::Test::requireApprox(columnResult.asDense(), matrix.toDense() * (column + column));
	Hoppy::Test::requireApprox(rowResult.asDense(), row * matrix.toDense());

	Eigen::Map<const Eigen::Vector3d> mappedColumn(column.data());
	Eigen::Ref<const Eigen::RowVector3d> referencedRow(row);
	Hoppy::Test::requireApprox((matrix * mappedColumn).asDense(), matrix.toDense() * column);
	Hoppy::Test::requireApprox((referencedRow * matrix).asDense(), row * matrix.toDense());
}

TEST_CASE("one-by-one operand position determines result orientation")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1};
	matrix[0](0, 0) = 3.0;
	Eigen::Matrix<double, 1, 1> value;
	value(0, 0) = 2.0;
	const auto column = matrix * value;
	const auto row = value * matrix;
	static_assert(std::is_same_v<typename decltype(column)::Orientation, Hoppy::Column>);
	static_assert(std::is_same_v<typename decltype(row)::Orientation, Hoppy::Row>);
	REQUIRE(column[0](0) == 6.0);
	REQUIRE(row[0](0) == 6.0);
}

TEST_CASE("empty matvecs and diagonal block matrices are supported")
{
	Hoppy::BlockDiagonalMatrix<double> emptyMatrix;
	Hoppy::BlockVector<double> emptyColumn;
	Hoppy::BlockVector<double, Hoppy::Row> emptyRow;
	REQUIRE((emptyMatrix * emptyColumn).size() == 0);
	REQUIRE((emptyRow * emptyMatrix).size() == 0);

	Hoppy::BlockVector<double> diagonal{1, 2};
	Hoppy::BlockVector<double> vector{1, 2};
	diagonal.asDense() << 2.0, 3.0, 4.0;
	vector.asDense() << 5.0, 6.0, 7.0;
	const auto result = diagonal.asDiagonal() * vector;
	Hoppy::Test::requireApprox(result.asDense(), diagonal.asDense().cwiseProduct(vector.asDense()));
}

TEST_CASE("unsupported block-vector orientations are absent")
{
	using Matrix = Hoppy::BlockDiagonalMatrix<double>;
	using Column = Hoppy::BlockVector<double, Hoppy::Column>;
	using Row = Hoppy::BlockVector<double, Hoppy::Row>;
	static_assert(has_product<Matrix, Column>::value);
	static_assert(has_product<Row, Matrix>::value);
	static_assert(!has_product<Matrix, Row>::value);
	static_assert(!has_product<Column, Matrix>::value);
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("matvec operands must match dimensions and blocking")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	Hoppy::BlockVector<double> wrongBlocking{3};
	Hoppy::BlockVector<double, Hoppy::Row> wrongRowBlocking{3};
	REQUIRE_THROWS_AS((void)(matrix * wrongBlocking), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS((void)(wrongRowBlocking * matrix), Hoppy::Test::EigenAssertionFailure);

	Eigen::Vector2d shortColumn;
	Eigen::Matrix<double, 1, 2> shortRow;
	REQUIRE_THROWS_AS((void)(matrix * shortColumn), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS((void)(shortRow * matrix), Hoppy::Test::EigenAssertionFailure);
}
#endif
