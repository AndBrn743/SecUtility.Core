// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <complex>
#include <memory>
#include <type_traits>
#include <utility>

namespace
{
	template <typename T, typename = void>
	struct has_rvalue_transpose : std::false_type
	{};

	template <typename T>
	struct has_rvalue_transpose<T, std::void_t<decltype(std::declval<T&&>().transpose())>> : std::true_type
	{};

	template <typename T, typename = void>
	struct has_rvalue_divide : std::false_type
	{};

	template <typename T>
	struct has_rvalue_divide<T, std::void_t<decltype(std::declval<T&&>() / 2.0)>> : std::true_type
	{};

	template <typename T, typename = void>
	struct has_rvalue_cast : std::false_type
	{};

	template <typename T>
	struct has_rvalue_cast<T, std::void_t<decltype(std::declval<T&&>().template cast<float>())>>
	    : std::true_type
	{};

	template <typename T>
	void fill(T& value, const typename T::Scalar offset = typename T::Scalar{})
	{
		for (Eigen::Index index = 0; index < value.storedSize(); ++index)
			value.data()[index] = static_cast<typename T::Scalar>(index + 1) + offset;
	}

	Hoppy::BlockDiagonalMatrix<double> makeMatrix()
	{
		Hoppy::BlockDiagonalMatrix<double> result{1, 2};
		fill(result);
		return result;
	}

	Hoppy::BlockVector<double> makeVector()
	{
		Hoppy::BlockVector<double> result{1, 2};
		fill(result);
		return result;
	}
}

TEST_CASE("lazy operators accept owning and expression rvalues")
{
	using Matrix = Hoppy::BlockDiagonalMatrix<double>;
	using Vector = Hoppy::BlockVector<double>;
	using MatrixUnary = decltype(std::declval<const Matrix&>().transpose());
	using MatrixBinary = decltype(std::declval<const Matrix&>() + std::declval<const Matrix&>());
	using VectorUnary = decltype(std::declval<const Vector&>().transpose());
	using VectorBinary = decltype(std::declval<const Vector&>() + std::declval<const Vector&>());

	static_assert(has_rvalue_transpose<Matrix>::value);
	static_assert(has_rvalue_transpose<Vector>::value);
	static_assert(has_rvalue_divide<Matrix>::value);
	static_assert(has_rvalue_divide<Vector>::value);
	static_assert(has_rvalue_cast<Matrix>::value);
	static_assert(has_rvalue_cast<Vector>::value);

	static_assert(has_rvalue_transpose<MatrixUnary>::value);
	static_assert(has_rvalue_transpose<MatrixBinary>::value);
	static_assert(has_rvalue_transpose<VectorUnary>::value);
	static_assert(has_rvalue_transpose<VectorBinary>::value);
	static_assert(has_rvalue_divide<MatrixUnary>::value);
	static_assert(has_rvalue_divide<MatrixBinary>::value);
	static_assert(has_rvalue_divide<VectorUnary>::value);
	static_assert(has_rvalue_divide<VectorBinary>::value);
	static_assert(has_rvalue_cast<MatrixUnary>::value);
	static_assert(has_rvalue_cast<MatrixBinary>::value);
	static_assert(has_rvalue_cast<VectorUnary>::value);
	static_assert(has_rvalue_cast<VectorBinary>::value);
}

TEST_CASE("owning-rvalue expressions can be evaluated within their full expression")
{
	const auto matrixDense = makeMatrix().toDense();
	const auto vectorDense = makeVector().toDense();

	Hoppy::Test::requireApprox((makeMatrix().transpose() / 2.0).eval().toDense(),
	                           (matrixDense.transpose() / 2.0).eval());
	Hoppy::Test::requireApprox((makeMatrix() + makeMatrix()).eval().toDense(),
	                           (matrixDense + matrixDense).eval());
	Hoppy::Test::requireApprox(makeMatrix().template cast<float>().eval().toDense(),
	                           matrixDense.cast<float>());
	Hoppy::Test::requireApprox((makeVector().transpose() / 2.0).eval().toDense(),
	                           (vectorDense.transpose() / 2.0).eval());
	Hoppy::Test::requireApprox((makeVector() + makeVector()).eval().toDense(),
	                           (vectorDense + vectorDense).eval());
}

TEST_CASE("blocking information is referenced from lvalues and copied from rvalues")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	const auto& referenced = matrix.blockingInfo();
	REQUIRE(std::addressof(referenced) == std::addressof(matrix.blockingInfo()));

	const auto owningTemporary = makeMatrix().blockingInfo();
	const auto expressionTemporary = (makeMatrix() + makeMatrix()).blockingInfo();
	REQUIRE(owningTemporary == (std::vector<Eigen::Index>{1, 2}));
	REQUIRE(expressionTemporary == (std::vector<Eigen::Index>{1, 2}));
}

TEST_CASE("unary expression prvalues can be consumed by scalar and unary operators")
{
	using Complex = std::complex<double>;
	Hoppy::BlockDiagonalMatrix<Complex> matrix{1, 2};
	fill(matrix, Complex{1.0, -0.5});
	const auto dense = matrix.toDense();

	const auto scaledTranspose = matrix.transpose() / 2.0;
	const auto chainedUnary = -matrix.conjugate().transpose();
	const auto castTranspose = matrix.transpose().template cast<std::complex<float>>();

	Hoppy::Test::requireApprox(scaledTranspose.toDense(), dense.transpose() / 2.0);
	Hoppy::Test::requireApprox(chainedUnary.toDense(), -dense.conjugate().transpose());
	Hoppy::Test::requireApprox(castTranspose.toDense(),
	                           dense.transpose().template cast<std::complex<float>>());

	// An expression lvalue must continue to use the inherited const& API.
	const auto transposeExpression = matrix.transpose();
	Hoppy::Test::requireApprox((transposeExpression / 4.0).toDense(), dense.transpose() / 4.0);
	Hoppy::Test::requireApprox(transposeExpression.conjugate().toDense(), dense.transpose().conjugate());
}

TEST_CASE("owning lvalues use the CRTP scalar multiply and divide overloads")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	fill(matrix);
	const auto matrixProduct = matrix * 3.0;
	const auto matrixQuotient = matrix / 2.0;
	matrix[1](0, 0) = 19.0;
	Hoppy::Test::requireApprox(matrixProduct.toDense(), matrix.toDense() * 3.0);
	Hoppy::Test::requireApprox(matrixQuotient.toDense(), matrix.toDense() / 2.0);

	Hoppy::BlockVector<double> column{1, 2};
	fill(column);
	const auto columnProduct = column * 4.0;
	const auto columnQuotient = column / 2.0;
	column[1](0) = 23.0;
	Hoppy::Test::requireApprox(columnProduct.toDense(), column.toDense() * 4.0);
	Hoppy::Test::requireApprox(columnQuotient.toDense(), column.toDense() / 2.0);

	Hoppy::BlockVector<double, Hoppy::Row> row{1, 2};
	fill(row);
	const auto rowProduct = row * 5.0;
	const auto rowQuotient = row / 4.0;
	row[1](1) = 29.0;
	using ProductOrientation = typename Eigen::internal::traits<decltype(rowProduct)>::Orientation;
	using QuotientOrientation = typename Eigen::internal::traits<decltype(rowQuotient)>::Orientation;
	static_assert(std::is_same_v<ProductOrientation, Hoppy::Row>);
	static_assert(std::is_same_v<QuotientOrientation, Hoppy::Row>);
	Hoppy::Test::requireApprox(rowProduct.toDense(), row.toDense() * 5.0);
	Hoppy::Test::requireApprox(rowQuotient.toDense(), row.toDense() / 4.0);
}

TEST_CASE("binary expression prvalues can occur in either operand position and be stored")
{
	Hoppy::BlockDiagonalMatrix<double> lhs{1, 2};
	Hoppy::BlockDiagonalMatrix<double> rhs{1, 2};
	fill(lhs);
	fill(rhs, 3.0);
	const auto lhsDense = lhs.toDense();
	const auto rhsDense = rhs.toDense();

	const auto leftNested = (lhs + rhs) - lhs;
	const auto rightNested = lhs + (rhs - lhs);
	const auto transposed = (lhs + rhs).transpose() / 2.0;
	const auto leftScaled = 3.0 * (lhs + rhs);
	const auto rightScaled = (lhs + rhs) * 3.0;
	const auto cast = (lhs - rhs).template cast<float>();

	Hoppy::Test::requireApprox(leftNested.toDense(), (lhsDense + rhsDense) - lhsDense);
	Hoppy::Test::requireApprox(rightNested.toDense(), lhsDense + (rhsDense - lhsDense));
	Hoppy::Test::requireApprox(transposed.toDense(), (lhsDense + rhsDense).transpose() / 2.0);
	Hoppy::Test::requireApprox(leftScaled.toDense(), 3.0 * (lhsDense + rhsDense));
	Hoppy::Test::requireApprox(rightScaled.toDense(), (lhsDense + rhsDense) * 3.0);
	Hoppy::Test::requireApprox(cast.toDense(), (lhsDense - rhsDense).cast<float>());

	REQUIRE(((lhs + rhs) / 2.0).sum() == ((lhsDense + rhsDense) / 2.0).sum());
}

TEST_CASE("block-vector prvalue chains preserve orientation and nested lifetimes")
{
	Hoppy::BlockVector<double> lhs{1, 2};
	Hoppy::BlockVector<double> rhs{1, 2};
	fill(lhs);
	fill(rhs, 2.0);
	const auto lhsDense = lhs.toDense();
	const auto rhsDense = rhs.toDense();

	const auto columnChain = ((lhs + rhs) / 2.0).transpose().transpose() - lhs;
	const auto rowChain = (lhs + rhs).transpose() / 2.0;
	using ColumnOrientation = typename Eigen::internal::traits<decltype(columnChain)>::Orientation;
	using RowOrientation = typename Eigen::internal::traits<decltype(rowChain)>::Orientation;
	static_assert(std::is_same_v<ColumnOrientation, Hoppy::Column>);
	static_assert(std::is_same_v<RowOrientation, Hoppy::Row>);

	Hoppy::Test::requireApprox(columnChain.toDense(), ((lhsDense + rhsDense) / 2.0) - lhsDense);
	Hoppy::Test::requireApprox(rowChain.toDense(), (lhsDense + rhsDense).transpose() / 2.0);
	REQUIRE(((lhs + rhs) / 2.0).isApprox((lhs + rhs) / 2.0));

	Hoppy::BlockVector<double, Hoppy::Row> rowLhs{1, 2};
	Hoppy::BlockVector<double, Hoppy::Row> rowRhs{1, 2};
	fill(rowLhs);
	fill(rowRhs, 2.0);
	const auto storedRowExpression = (rowLhs + rowRhs).conjugate() / 2.0;
	Hoppy::Test::requireApprox(storedRowExpression.toDense(),
	                           (rowLhs.toDense() + rowRhs.toDense()).conjugate() / 2.0);
}
