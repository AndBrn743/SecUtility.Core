// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/BlockPolicy.hpp>
#include <SecUtility/Hoppy/BlockDiagonalMatrix.hpp>
#include <SecUtility/Hoppy/BlockVector.hpp>
#include <SecUtility/Hoppy/Detail/CheckedDimensions.hpp>
#include <SecUtility/Hoppy/Detail/Traits.hpp>

#include <Eigen/Core>

#include <cmath>
#include <cstdint>
#include <forward_list>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
	template <typename Iterator, typename = void>
	struct can_build_dimensions : std::false_type
	{};

	template <typename Iterator>
	struct can_build_dimensions<
	        Iterator,
	        std::void_t<decltype(Hoppy::Detail::BuildCheckedDimensions(
	                std::declval<Iterator>(), std::declval<Iterator>()))>> : std::true_type
	{};

	template <typename T, typename = void>
	struct has_two_argument_resize : std::false_type
	{};

	template <typename T>
	struct has_two_argument_resize<T, std::void_t<decltype(std::declval<T&>().resize(1, 2))>> : std::true_type
	{};

	class CountingForwardIterator
	{
	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = int;
		using difference_type = std::ptrdiff_t;
		using pointer = const int*;
		using reference = int;

		CountingForwardIterator() = default;
		CountingForwardIterator(const int* current, int* dereferences)
			: m_Current(current), m_Dereferences(dereferences)
		{}

		int operator*() const
		{
			++*m_Dereferences;
			return *m_Current;
		}

		CountingForwardIterator& operator++()
		{
			++m_Current;
			return *this;
		}

		CountingForwardIterator operator++(int)
		{
			auto copy = *this;
			++*this;
			return copy;
		}

		friend bool operator==(const CountingForwardIterator& lhs, const CountingForwardIterator& rhs)
		{
			return lhs.m_Current == rhs.m_Current;
		}

		friend bool operator!=(const CountingForwardIterator& lhs, const CountingForwardIterator& rhs)
		{
			return !(lhs == rhs);
		}

	private:
		const int* m_Current = nullptr;
		int* m_Dereferences = nullptr;
	};

	struct NonIntegral
	{};

	static_assert(Hoppy::IsBlockPolicy<Hoppy::DenseBlockPolicy>);
	static_assert(Hoppy::is_block_policy<Hoppy::DenseBlockPolicy>::value);
	static_assert(!Hoppy::IsBlockPolicy<const Hoppy::DenseBlockPolicy>);
	static_assert(!Hoppy::IsBlockPolicy<NonIntegral>);
	static_assert(std::is_same_v<Hoppy::DenseBlockPolicy::View<double>,
	                             Eigen::Map<Eigen::MatrixX<double>, Eigen::Unaligned>>);
	static_assert(std::is_same_v<Hoppy::DenseBlockPolicy::ConstView<double>,
	                             Eigen::Map<const Eigen::MatrixX<double>, Eigen::Unaligned>>);
	static_assert(std::is_same_v<decltype(Hoppy::DenseBlockPolicy::BlockElementCount(Eigen::Index{2})),
	                             Eigen::Index>);
	static_assert(can_build_dimensions<std::vector<int>::const_iterator>::value);
	static_assert(can_build_dimensions<std::forward_list<unsigned>::const_iterator>::value);
	static_assert(!can_build_dimensions<std::vector<bool>::const_iterator>::value);
	static_assert(!can_build_dimensions<std::vector<double>::const_iterator>::value);
	static_assert(!can_build_dimensions<std::vector<NonIntegral>::const_iterator>::value);
	static_assert(!can_build_dimensions<std::istream_iterator<int>>::value);
	static_assert(Eigen::internal::traits<Hoppy::BlockDiagonalMatrix<double>>::RowsAtCompileTime
	              == Eigen::Dynamic);
	static_assert(Eigen::internal::traits<Hoppy::BlockDiagonalMatrix<double>>::ColsAtCompileTime
	              == Eigen::Dynamic);
	static_assert(Eigen::internal::traits<Hoppy::BlockVector<double, Hoppy::Column>>::RowsAtCompileTime
	              == Eigen::Dynamic);
	static_assert(Eigen::internal::traits<Hoppy::BlockVector<double, Hoppy::Column>>::ColsAtCompileTime == 1);
	static_assert(Eigen::internal::traits<Hoppy::BlockVector<double, Hoppy::Row>>::RowsAtCompileTime == 1);
	static_assert(Eigen::internal::traits<Hoppy::BlockVector<double, Hoppy::Row>>::ColsAtCompileTime
	              == Eigen::Dynamic);
	static_assert(!std::is_same_v<Hoppy::Column, Hoppy::Row>);
	static_assert(!has_two_argument_resize<Hoppy::BlockDiagonalMatrix<double>>::value);
	static_assert(!has_two_argument_resize<Hoppy::BlockVector<double>>::value);
}

TEST_CASE("DenseBlockPolicy reports dense square storage and maps")
{
	REQUIRE(Hoppy::DenseBlockPolicy::BlockElementCount(3) == 9);
	double values[]{1.0, 2.0, 3.0, 4.0};
	Hoppy::DenseBlockPolicy::View<double> view(values, 2, 2);
	REQUIRE(view(0, 0) == 1.0);
	REQUIRE(view(1, 0) == 2.0);
	REQUIRE(view(0, 1) == 3.0);
}

TEST_CASE("checked dimensions build exact logical and storage offsets")
{
	const std::vector<int> empty;
	const auto emptyResult = Hoppy::Detail::BuildCheckedDimensions(empty.begin(), empty.end());
	REQUIRE(emptyResult.Dimensions.empty());
	REQUIRE(emptyResult.BlockOffsets == std::vector<Eigen::Index>{0});
	REQUIRE(emptyResult.StorageOffsets == std::vector<Eigen::Index>{0});
	REQUIRE(emptyResult.TotalDimension == 0);
	REQUIRE(emptyResult.StoredSize == 0);

	const std::vector<short> single{3};
	const auto singleResult = Hoppy::Detail::BuildCheckedDimensions(single.begin(), single.end());
	REQUIRE(singleResult.Dimensions == std::vector<Eigen::Index>{3});
	REQUIRE(singleResult.BlockOffsets == std::vector<Eigen::Index>{0, 3});
	REQUIRE(singleResult.StorageOffsets == std::vector<Eigen::Index>{0, 9});

	const std::forward_list<unsigned> dimensions{1, 2, 3};
	const auto result = Hoppy::Detail::BuildCheckedDimensions(dimensions.begin(), dimensions.end());
	REQUIRE(result.Dimensions == std::vector<Eigen::Index>{1, 2, 3});
	REQUIRE(result.BlockOffsets == std::vector<Eigen::Index>{0, 1, 3, 6});
	REQUIRE(result.StorageOffsets == std::vector<Eigen::Index>{0, 1, 5, 14});
	REQUIRE(result.TotalDimension == 6);
	REQUIRE(result.StoredSize == 14);

	const auto vectorResult = Hoppy::Detail::BuildCheckedVectorDimensions(dimensions.begin(), dimensions.end());
	REQUIRE(vectorResult.Dimensions == std::vector<Eigen::Index>{1, 2, 3});
	REQUIRE(vectorResult.BlockOffsets == std::vector<Eigen::Index>{0, 1, 3, 6});
	REQUIRE(vectorResult.StorageOffsets == std::vector<Eigen::Index>{0, 1, 3, 6});
	REQUIRE(vectorResult.TotalDimension == 6);
	REQUIRE(vectorResult.StoredSize == 6);
}

TEST_CASE("checked dimensions dereference each input exactly once")
{
	const int dimensions[]{1, 2, 3};
	int dereferences = 0;
	const auto result = Hoppy::Detail::BuildCheckedDimensions(
	        CountingForwardIterator(dimensions, &dereferences),
	        CountingForwardIterator(dimensions + 3, &dereferences));
	REQUIRE(result.TotalDimension == 6);
	REQUIRE(dereferences == 3);
}

TEST_CASE("dimension conversion accepts exact signed and unsigned boundaries")
{
	Eigen::Index converted = 0;
	const auto maximum = std::numeric_limits<Eigen::Index>::max();
	REQUIRE(Hoppy::Detail::CheckedDimensionsDetail::tryConvertPositive(maximum, converted));
	REQUIRE(converted == maximum);
	REQUIRE(Hoppy::Detail::CheckedDimensionsDetail::tryConvertPositive(std::uint32_t{7}, converted));
	REQUIRE(converted == 7);
	REQUIRE_FALSE(Hoppy::Detail::CheckedDimensionsDetail::tryConvertPositive(
	        std::numeric_limits<std::uint64_t>::max(), converted));
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("checked dimensions reject invalid values and every independent overflow")
{
	const auto maximum = std::numeric_limits<Eigen::Index>::max();
	const auto expectRejected = [](const auto& dimensions) {
		REQUIRE_THROWS_AS(
		        Hoppy::Detail::BuildCheckedDimensions(dimensions.begin(), dimensions.end()),
		        Hoppy::Test::EigenAssertionFailure);
	};

	expectRejected(std::vector<int>{0});
	expectRejected(std::vector<int>{-1});
	expectRejected(std::vector<std::uint64_t>{std::numeric_limits<std::uint64_t>::max()});
	Eigen::Index ignored = 0;
	REQUIRE_FALSE(Hoppy::Detail::CheckedDimensionsDetail::checkedAdd(maximum, 1, ignored));

	Eigen::Index largestSquareRoot = static_cast<Eigen::Index>(std::sqrt(static_cast<long double>(maximum)));
	while (largestSquareRoot < maximum / largestSquareRoot)
		++largestSquareRoot;
	while (largestSquareRoot > maximum / largestSquareRoot)
		--largestSquareRoot;
	const Eigen::Index squareOverflow = largestSquareRoot + 1;
	expectRejected(std::vector<Eigen::Index>{squareOverflow});

	const Eigen::Index largeValidBlock = largestSquareRoot;
	expectRejected(std::vector<Eigen::Index>{largeValidBlock, largeValidBlock, largeValidBlock});
	expectRejected(std::vector<Eigen::Index>{largeValidBlock, 1});
}
#endif

TEST_CASE("owning factories and fills preserve exact specialization and blocking")
{
	using Matrix = Hoppy::BlockDiagonalMatrix<double>;
	using RowVector = Hoppy::BlockVector<double, Hoppy::Row>;
	const std::vector<Eigen::Index> dimensions{1, 2, 3};

	const auto zero = Matrix::Zero(dimensions);
	const auto ones = Matrix::Ones({1, 2, 3});
	const auto constant = Matrix::Constant(dimensions.begin(), dimensions.end(), 4.0);
	const auto identity = Matrix::Identity(dimensions);
	static_assert(std::is_same_v<std::remove_const_t<decltype(zero)>, Matrix>);
	REQUIRE(zero.data()[0] == 0.0);
	REQUIRE(ones[2].isOnes());
	REQUIRE(constant[1].isConstant(4.0));
	REQUIRE(identity[2].isIdentity());

	Matrix matrix{3};
	matrix.setZero(dimensions).setOnes();
	REQUIRE(matrix.blockingInfo() == dimensions);
	REQUIRE(matrix[0].isOnes());
	matrix.setIdentity({2, 1});
	REQUIRE(matrix.blockingInfo() == std::vector<Eigen::Index>{2, 1});
	REQUIRE(matrix[0].isIdentity());

	const auto row = RowVector::Constant({1, 2, 3}, 7.0);
	static_assert(std::is_same_v<std::remove_const_t<decltype(row)>, RowVector>);
	REQUIRE(row.blockingInfo() == dimensions);
	REQUIRE(row.asDense().isConstant(7.0));
}

TEST_CASE("FromDense and FromDenseLike extract exact blocks and vector orientation")
{
	using Matrix = Hoppy::BlockDiagonalMatrix<double>;
	using ColumnVector = Hoppy::BlockVector<double, Hoppy::Column>;
	using RowVector = Hoppy::BlockVector<double, Hoppy::Row>;
	Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(3, 3);
	dense.diagonal() << 1.0, 2.0, 3.0;
	dense(1, 2) = 4.0;
	dense(2, 1) = 5.0;
	const auto matrix = Matrix::FromDense(dense, {1, 2});
	REQUIRE(matrix[0](0, 0) == 1.0);
	REQUIRE(matrix[1].isApprox(dense.bottomRightCorner(2, 2)));
	const auto like = Matrix::FromDenseLike(dense, matrix);
	REQUIRE(like[1].isApprox(matrix[1]));

	Eigen::RowVectorXd rowDense(3);
	rowDense << 6.0, 7.0, 8.0;
	const auto column = ColumnVector::FromDense(rowDense, {1, 2});
	const auto row = RowVector::FromDenseLike(column.asDense(), matrix);
	REQUIRE(column.asDense().isApprox(rowDense.transpose()));
	REQUIRE(row.asDense().isApprox(rowDense));
}

TEST_CASE("FromBlocks and SingleBlock infer blocking and convert scalar values")
{
	using Matrix = Hoppy::BlockDiagonalMatrix<double>;
	std::vector<Eigen::MatrixXf> blocks;
	blocks.emplace_back(Eigen::MatrixXf::Constant(1, 1, 2.0F));
	blocks.emplace_back(Eigen::MatrixXf::Constant(2, 2, 3.0F));
	const auto matrix = Matrix::FromBlocks(blocks.begin(), blocks.end());
	REQUIRE(matrix.blockingInfo() == std::vector<Eigen::Index>{1, 2});
	REQUIRE(matrix[1].isConstant(3.0));

	Eigen::MatrixXd source = Eigen::MatrixXd::Identity(2, 2);
	const auto single = Matrix::SingleBlock(source + source);
	REQUIRE(single.blockingInfo() == std::vector<Eigen::Index>{2});
	REQUIRE(single[0].isApprox(2.0 * source));

	std::vector<Eigen::VectorXf> vectors;
	vectors.emplace_back(Eigen::VectorXf::Constant(1, 4.0F));
	vectors.emplace_back(Eigen::VectorXf::Constant(2, 5.0F));
	const auto vector = Hoppy::BlockVector<double>::FromBlocks(vectors.begin(), vectors.end());
	REQUIRE(vector.blockingInfo() == std::vector<Eigen::Index>{1, 2});
	REQUIRE(vector.asDense().isApprox((Eigen::Vector3d() << 4.0, 5.0, 5.0).finished()));
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("reblocking and extraction failures leave the old object untouched")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	matrix.setConstant(9.0);
	REQUIRE_THROWS_AS(matrix.resize({1, 0}), Hoppy::Test::EigenAssertionFailure);
	REQUIRE(matrix.blockingInfo() == std::vector<Eigen::Index>{1, 2});
	REQUIRE(matrix[1].isConstant(9.0));

	Eigen::MatrixXd wrong = Eigen::MatrixXd::Zero(2, 3);
	REQUIRE_THROWS_AS((Hoppy::BlockDiagonalMatrix<double>::FromDense(wrong, {2})),
	                  Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS((Hoppy::BlockVector<double>::SingleBlock(wrong)),
	                  Hoppy::Test::EigenAssertionFailure);
}
#endif
