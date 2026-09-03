// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <type_traits>

namespace Hoppy::Test
{
	template <typename Source>
	class PassThrough;
}

namespace Eigen::internal
{
	template <typename Source>
	struct traits<Hoppy::Test::PassThrough<Source>> : traits<Source>
	{};
}

namespace Hoppy::Test
{
	template <typename Source>
	class PassThrough : public std::conditional_t<
	                            std::is_same_v<typename Eigen::internal::traits<Source>::StorageKind,
	                                           Hoppy::Detail::BlockDiagonalStorage>,
	                            Hoppy::BlockDiagonalMatrixExpr<PassThrough<Source>>,
	                            Hoppy::BlockVectorExpr<PassThrough<Source>>>
	{
	public:
		using Scalar = typename Source::Scalar;
		explicit PassThrough(const Source& source) : m_Source(source) {}
		Eigen::Index blockCount() const { return m_Source.blockCount(); }
		Eigen::Index rows() const { return m_Source.rows(); }
		Eigen::Index cols() const { return m_Source.cols(); }
		Eigen::Index size() const { return m_Source.size(); }
		Eigen::Index totalDimension() const { return m_Source.totalDimension(); }
		Eigen::Index storedSize() const { return m_Source.storedSize(); }
		Eigen::Index dimensionOfBlock(Eigen::Index i) const { return m_Source.dimensionOfBlock(i); }
		Eigen::Index blockOffset(Eigen::Index i) const { return m_Source.blockOffset(i); }
		Eigen::Index storageOffset(Eigen::Index i) const { return m_Source.storageOffset(i); }
		std::vector<Eigen::Index> blockingInfo() const { return m_Source.blockingInfo(); }
		auto operator[](Eigen::Index i) const { return m_Source[i]; }
		const Scalar* data() const { return m_Source.data(); }

	private:
		const Source& m_Source;
	};

	template <typename T, typename = void>
	struct has_arrow : std::false_type
	{};
	template <typename T>
	struct has_arrow<T, std::void_t<decltype(std::declval<T>().operator->())>> : std::true_type
	{};

	static_assert(!noexcept(std::declval<const PassThrough<Hoppy::BlockDiagonalMatrix<double>>&>().rows()));
	static_assert(!noexcept(std::declval<const PassThrough<Hoppy::BlockDiagonalMatrix<double>>&>().blockCount()));
}

TEST_CASE("block expression protocol iterates proxy blocks and compares blocking")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2, 3};
	matrix.setConstant(2.0);
	Hoppy::Test::PassThrough expression(matrix);
	REQUIRE(expression.hasSameBlockingAs(matrix));
	Eigen::Index index = 0;
	for (const auto block : expression)
	{
		REQUIRE(block.rows() == matrix.dimensionOfBlock(index));
		REQUIRE(block.isConstant(2.0));
		++index;
	}
	REQUIRE(index == 3);
	static_assert(!Hoppy::Test::has_arrow<decltype(expression.begin())>::value);
}

TEST_CASE("BD expressions evaluate to plain and structurally dense results")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	matrix[0].setConstant(3.0);
	matrix[1] << 1.0, 2.0, 4.0, 5.0;
	Hoppy::Test::PassThrough expression(matrix);
	const auto plain = expression.eval();
	static_assert(std::is_same_v<std::remove_const_t<decltype(plain)>, Hoppy::BlockDiagonalMatrix<double>>);
	REQUIRE(plain[1].isApprox(matrix[1]));
	const auto dense = expression.toDense();
	REQUIRE(dense.topLeftCorner(1, 1)(0, 0) == 3.0);
	REQUIRE(dense.bottomRightCorner(2, 2).isApprox(matrix[1]));
	REQUIRE(dense.topRightCorner(1, 2).isZero());

	Hoppy::BlockDiagonalMatrix<double> assigned{3};
	assigned = expression;
	REQUIRE(assigned.blockingInfo() == matrix.blockingInfo());
	REQUIRE(assigned[1].isApprox(matrix[1]));
}

TEST_CASE("oriented BV expressions evaluate and resize dense destinations")
{
	Hoppy::BlockVector<double, Hoppy::Row> row{1, 2};
	row.asDense() << 1.0, 2.0, 3.0;
	Hoppy::Test::PassThrough expression(row);
	const auto plain = expression.eval();
	static_assert(std::is_same_v<std::remove_const_t<decltype(plain)>,
	                             Hoppy::BlockVector<double, Hoppy::Row>>);
	REQUIRE(plain.asDense().isApprox(row.asDense()));
	Eigen::MatrixXd destination;
	expression.evalTo(destination);
	REQUIRE(destination.rows() == 1);
	REQUIRE(destination.cols() == 3);
	REQUIRE(destination.isApprox(row.asDense()));

	Hoppy::BlockVector<double, Hoppy::Column> column{1, 2};
	column.asDense() << 4.0, 5.0, 6.0;
	Hoppy::Test::PassThrough columnExpression(column);
	Eigen::MatrixXd columnDestination;
	columnExpression.evalTo(columnDestination);
	REQUIRE(columnDestination.rows() == 3);
	REQUIRE(columnDestination.cols() == 1);
	REQUIRE(columnDestination.isApprox(column.asDense()));
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("evalTo rejects destination overlap and fixed-size shape mismatch")
{
	Hoppy::BlockVector<double> vector{1, 2};
	Hoppy::Test::PassThrough expression(vector);
	auto overlapping = vector.asDense();
	REQUIRE_THROWS_AS(expression.evalTo(overlapping), Hoppy::Test::EigenAssertionFailure);

	Eigen::Matrix<double, 2, 2> wrongSize;
	REQUIRE_THROWS_AS(expression.evalTo(wrongSize), Hoppy::Test::EigenAssertionFailure);
}
#endif
