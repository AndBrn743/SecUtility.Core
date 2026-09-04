// SPDX-License-Identifier: MIT

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/Hoppy.hpp>

#include <Eigen/Core>

#include <complex>
#include <type_traits>
#include <utility>

namespace
{
	template <typename Matrix>
	void fill(Matrix& matrix)
	{
		for (Eigen::Index index = 0; index < matrix.storedSize(); ++index)
			matrix.data()[index] = static_cast<typename Matrix::Scalar>(index + 1);
	}
}

TEST_CASE("block diagonal times dense matrix uses an Eigen dense product")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2, 3};
	fill(matrix);
	Eigen::Matrix<double, 6, 4> rhs;
	rhs.setRandom();
	const auto product = matrix * rhs;
	using Product = std::remove_cv_t<decltype(product)>;
	static_assert(std::is_same_v<Product,
	                             Eigen::Product<Hoppy::BlockDiagonalMatrix<double>, decltype(rhs)>>);
	Hoppy::Test::requireApprox(product.eval(), matrix.toDense() * rhs);

	Eigen::MatrixXd accumulated = Eigen::MatrixXd::Ones(6, 4);
	accumulated += product;
	Hoppy::Test::requireApprox(accumulated, Eigen::MatrixXd::Ones(6, 4) + matrix.toDense() * rhs);
	Eigen::MatrixXd scaled = 2.5 * product;
	Hoppy::Test::requireApprox(scaled, 2.5 * (matrix.toDense() * rhs));
}

TEST_CASE("dense matrix times block diagonal supports rectangular and row-major operands")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	fill(matrix);
	Eigen::Matrix<double, 4, 3, Eigen::RowMajor> lhs;
	lhs.setRandom();
	const auto product = lhs * matrix;
	static_assert(std::is_same_v<std::remove_cv_t<decltype(product)>,
	                             Eigen::Product<decltype(lhs), Hoppy::BlockDiagonalMatrix<double>>>);
	Hoppy::Test::requireApprox(product.eval(), lhs * matrix.toDense());

	Eigen::MatrixXd accumulated = Eigen::MatrixXd::Ones(4, 3);
	accumulated -= product;
	Hoppy::Test::requireApprox(accumulated, Eigen::MatrixXd::Ones(4, 3) - lhs * matrix.toDense());
}

TEST_CASE("mapped strided blocked and lazy dense operands remain expressions")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	fill(matrix);
	Eigen::Matrix<double, 3, 6> storage;
	storage.setRandom();
	const auto rhsBlock = storage.block<3, 2>(0, 1);
	const auto lhsBlock = storage.block<2, 3>(0, 0);
	Hoppy::Test::requireApprox((matrix * (rhsBlock + rhsBlock)).eval(),
	                           matrix.toDense() * (rhsBlock + rhsBlock));
	Hoppy::Test::requireApprox(((lhsBlock + lhsBlock) * matrix).eval(),
	                           (lhsBlock + lhsBlock) * matrix.toDense());

	Eigen::Matrix<double, 6, 2> mappedStorage;
	mappedStorage.setRandom();
	Eigen::Map<const Eigen::Matrix<double, 3, 2>, 0, Eigen::Stride<6, 1>> strided(
	        mappedStorage.data(), Eigen::Stride<6, 1>{});
	Hoppy::Test::requireApprox((matrix * strided).eval(), matrix.toDense() * strided);
}

TEST_CASE("dynamic runtime vectors use dense products while fixed vectors remain block vectors")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	fill(matrix);
	Eigen::MatrixXd dynamicColumn(3, 1);
	dynamicColumn << 1.0, 2.0, 3.0;
	Eigen::MatrixXd dynamicRow(1, 3);
	dynamicRow << 1.0, 2.0, 3.0;
	const auto denseColumn = matrix * dynamicColumn;
	const auto denseRow = dynamicRow * matrix;
	static_assert(std::is_same_v<std::remove_cv_t<decltype(denseColumn)>,
	                             Eigen::Product<Hoppy::BlockDiagonalMatrix<double>, Eigen::MatrixXd>>);
	static_assert(std::is_same_v<std::remove_cv_t<decltype(denseRow)>,
	                             Eigen::Product<Eigen::MatrixXd, Hoppy::BlockDiagonalMatrix<double>>>);
	Hoppy::Test::requireApprox(denseColumn.eval(), matrix.toDense() * dynamicColumn);
	Hoppy::Test::requireApprox(denseRow.eval(), dynamicRow * matrix.toDense());

	Eigen::Vector3d fixedColumn = dynamicColumn;
	Eigen::RowVector3d fixedRow = dynamicRow;
	static_assert(std::is_same_v<decltype(matrix * fixedColumn), Hoppy::BlockVector<double>>);
	static_assert(std::is_same_v<decltype(fixedRow * matrix), Hoppy::BlockVector<double, Hoppy::Row>>);
}

TEST_CASE("empty dimensions and zero-column products are supported")
{
	Hoppy::BlockDiagonalMatrix<double> empty;
	Eigen::MatrixXd rhs(0, 3);
	Eigen::MatrixXd lhs(4, 0);
	REQUIRE((empty * rhs).eval().rows() == 0);
	REQUIRE((empty * rhs).eval().cols() == 3);
	REQUIRE((lhs * empty).eval().rows() == 4);
	REQUIRE((lhs * empty).eval().cols() == 0);

	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	Eigen::MatrixXd noColumns(3, 0);
	const auto result = (matrix * noColumns).eval();
	REQUIRE(result.rows() == 3);
	REQUIRE(result.cols() == 0);
}

TEST_CASE("mixed real-complex products use Eigen product scalar traits")
{
	using Complex = std::complex<double>;
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	fill(matrix);
	Eigen::MatrixXcd rhs(3, 2);
	rhs.setConstant(Complex{1.0, 2.0});
	const auto product = matrix * rhs;
	static_assert(std::is_same_v<typename std::remove_cv_t<decltype(product)>::Scalar, Complex>);
	Hoppy::Test::requireApprox(product.eval(), matrix.toDense() * rhs);
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("dense products enforce inner dimensions")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	Eigen::MatrixXd wrongRhs(2, 2);
	Eigen::MatrixXd wrongLhs(2, 2);
	REQUIRE_THROWS_AS((void)(matrix * wrongRhs), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS((void)(wrongLhs * matrix), Hoppy::Test::EigenAssertionFailure);
}
#endif
