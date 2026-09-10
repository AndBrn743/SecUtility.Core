// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/BlockDiagonalMatrix.hpp>
#include <SecUtility/Hoppy/BlockVector.hpp>

#include <Eigen/Core>

#include <complex>
#include <forward_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
	using Matrix = Hoppy::BlockDiagonalMatrix<double>;
	using ColumnVector = Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Column>;
	using RowVector = Hoppy::BlockVector<double, Hoppy::BlockVectorOrientation::Row>;

	template <typename T>
	void accept_lvalue(T&);

	template <typename T>
	void copyAssign(T& destination, const T& source)
	{
		destination = source;
	}

	template <typename T, typename = void>
	struct can_bind_block_lvalue : std::false_type
	{};

	template <typename T>
	struct can_bind_block_lvalue<T, std::void_t<decltype(accept_lvalue(std::declval<T&>()[0]))>>
	    : std::true_type
	{};

	static_assert(std::is_same_v<decltype(std::declval<Matrix&>()[0]), Matrix::Block>);
	static_assert(std::is_same_v<decltype(std::declval<const Matrix&>()[0]), Matrix::ConstBlock>);
	static_assert(std::is_same_v<Matrix::Block,
	                             Eigen::Map<Eigen::MatrixX<double>, Eigen::Unaligned>>);
	static_assert(std::is_same_v<ColumnVector::Block,
	                             Eigen::Map<Eigen::VectorX<double>, Eigen::Unaligned>>);
	static_assert(std::is_same_v<RowVector::Block,
	                             Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic>, Eigen::Unaligned>>);
	static_assert(std::is_same_v<decltype(std::declval<const ColumnVector&>()[0]),
	                             ColumnVector::ConstBlock>);
	static_assert(std::is_same_v<decltype(std::declval<const RowVector&>()[0]),
	                             RowVector::ConstBlock>);
	static_assert(!can_bind_block_lvalue<Matrix>::value);

	static_assert(std::is_nothrow_move_constructible_v<Matrix>);
	static_assert(std::is_nothrow_move_assignable_v<Matrix>);
	static_assert(std::is_nothrow_swappable_v<Matrix>);
	static_assert(noexcept(std::declval<const Matrix&>().rows()));
	static_assert(noexcept(std::declval<const Matrix&>().cols()));
	static_assert(noexcept(std::declval<const Matrix&>().size()));
	static_assert(noexcept(std::declval<const Matrix&>().blockCount()));
	static_assert(noexcept(std::declval<const Matrix&>().totalDimension()));
	static_assert(noexcept(std::declval<const Matrix&>().storedSize()));
	static_assert(noexcept(std::declval<Matrix&>().data()));
	static_assert(std::is_same_v<decltype(std::declval<const Matrix&>().data()), const double*>);
	static_assert(!noexcept(std::declval<const Matrix&>().dimensionOfBlock(0)));
	static_assert(!noexcept(std::declval<const Matrix&>().blockOffset(0)));
	static_assert(!noexcept(std::declval<const Matrix&>().storageOffset(0)));
	static_assert(!noexcept(std::declval<Matrix&>()[0]));
}

TEST_CASE("owning objects expose exact empty and nonuniform metadata")
{
	const Matrix empty;
	REQUIRE(empty.blockCount() == 0);
	REQUIRE(empty.rows() == 0);
	REQUIRE(empty.cols() == 0);
	REQUIRE(empty.size() == 0);
	REQUIRE(empty.totalDimension() == 0);
	REQUIRE(empty.storedSize() == 0);
	REQUIRE(empty.blockingInfo().empty());
	REQUIRE(ColumnVector{}.asDense().size() == 0);
	REQUIRE(RowVector{}.asDense().size() == 0);

	const std::forward_list<unsigned> dimensions{1, 2, 3};
	const Matrix matrix(dimensions.begin(), dimensions.end());
	REQUIRE(matrix.blockingInfo() == std::vector<Eigen::Index>{1, 2, 3});
	REQUIRE(matrix.blockCount() == 3);
	REQUIRE(matrix.rows() == 6);
	REQUIRE(matrix.cols() == 6);
	REQUIRE(matrix.size() == 36);
	REQUIRE(matrix.totalDimension() == 6);
	REQUIRE(matrix.storedSize() == 14);
	REQUIRE(matrix.dimensionOfBlock(1) == 2);
	REQUIRE(matrix.blockOffset(2) == 3);
	REQUIRE(matrix.storageOffset(2) == 5);

	const ColumnVector column{1, 2, 3};
	REQUIRE(column.rows() == 6);
	REQUIRE(column.cols() == 1);
	REQUIRE(column.size() == 6);
	REQUIRE(column.storedSize() == 6);

	const RowVector row{1, 2, 3};
	REQUIRE(row.rows() == 1);
	REQUIRE(row.cols() == 6);
	REQUIRE(row.size() == 6);
}

TEST_CASE("owning buffers are value initialized and block maps alias contiguous storage")
{
	Hoppy::BlockDiagonalMatrix<std::complex<double>> matrix{1, 2, 3};
	for (Eigen::Index index = 0; index < matrix.storedSize(); ++index)
		REQUIRE(matrix.data()[index] == std::complex<double>{});

	Eigen::MatrixX<std::complex<double>> block(2, 2);
	block << std::complex<double>{1, 2}, std::complex<double>{3, 4},
	         std::complex<double>{5, 6}, std::complex<double>{7, 8};
	matrix[1] = block;
	REQUIRE(matrix[1].isApprox(block));
	for (Eigen::Index index = 0; index < 4; ++index)
		REQUIRE(matrix.data()[matrix.storageOffset(1) + index] == block.data()[index]);

	const auto& constMatrix = matrix;
	REQUIRE(constMatrix[1].isApprox(block));
}

TEST_CASE("BlockVector block and dense maps provide read-write aliasing in both orientations")
{
	ColumnVector column{1, 2};
	column.asDense() << 1.0, 2.0, 3.0;
	REQUIRE(column[0](0) == 1.0);
	REQUIRE(column[1](0) == 2.0);
	column[1](1) = 4.0;
	REQUIRE(column.asDense()(2) == 4.0);

	RowVector row{1, 2};
	row.asDense() << 5.0, 6.0, 7.0;
	REQUIRE(row[0](0) == 5.0);
	REQUIRE(row[1](1) == 7.0);
	row[1](0) = 8.0;
	REQUIRE(row.asDense()(1) == 8.0);

	const auto& constColumn = column;
	const auto& constRow = row;
	static_assert(std::is_same_v<decltype(constColumn.asDense()), ColumnVector::ConstBlock>);
	static_assert(std::is_same_v<decltype(constRow.asDense()), RowVector::ConstBlock>);
	REQUIRE(constColumn.asDense()(0) == 1.0);
	REQUIRE(constRow.asDense()(0) == 5.0);
	REQUIRE(constColumn[0](0) == 1.0);
	REQUIRE(constColumn[1](1) == 4.0);
	REQUIRE(constRow[0](0) == 5.0);
	REQUIRE(constRow[1](0) == 8.0);
}

TEST_CASE("copy, move, copy self-assignment, and ADL swap preserve container semantics")
{
	Matrix original{1, 2};
	original[0](0, 0) = 3.0;
	original[1].setConstant(4.0);

	Matrix copy = original;
	copy[0](0, 0) = 9.0;
	REQUIRE(original[0](0, 0) == 3.0);

	copyAssign(copy, copy);
	REQUIRE(copy[0](0, 0) == 9.0);

	Matrix moved = std::move(copy);
	REQUIRE(moved.blockingInfo() == std::vector<Eigen::Index>{1, 2});
	copy = Matrix{1};
	REQUIRE(copy.blockingInfo() == std::vector<Eigen::Index>{1});

	Matrix assigned{3};
	assigned = std::move(moved);
	REQUIRE(assigned.blockingInfo() == std::vector<Eigen::Index>{1, 2});
	moved = Matrix{2};
	REQUIRE(moved.blockingInfo() == std::vector<Eigen::Index>{2});

	using std::swap;
	swap(original, assigned);
	REQUIRE(original[0](0, 0) == 9.0);
	REQUIRE(assigned[0](0, 0) == 3.0);

	ColumnVector vector{1, 2};
	vector.asDense() << 1.0, 2.0, 3.0;
	ColumnVector movedVector = std::move(vector);
	REQUIRE(movedVector.asDense()(2) == 3.0);
	vector = ColumnVector{1};
	REQUIRE(vector.blockingInfo() == std::vector<Eigen::Index>{1});
}

#ifdef HOPPY_TEST_EIGEN_ASSERT_THROWS
TEST_CASE("block and offset access reject negative, terminal, and empty indices")
{
	Matrix matrix{1, 2};
	REQUIRE_THROWS_AS(matrix[-1], Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(matrix[2], Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(matrix.dimensionOfBlock(2), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(matrix.blockOffset(2), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(matrix.storageOffset(2), Hoppy::Test::EigenAssertionFailure);

	Matrix empty;
	REQUIRE_THROWS_AS(empty[0], Hoppy::Test::EigenAssertionFailure);
}
#endif
