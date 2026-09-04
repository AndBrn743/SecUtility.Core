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

	template <typename Dense, typename Block, typename = void>
	struct has_sum : std::false_type
	{};

	template <typename Dense, typename Block>
	struct has_sum<Dense, Block,
	               std::void_t<decltype(std::declval<Dense>() + std::declval<Block>())>>
	    : std::true_type
	{};
}  // namespace

TEST_CASE("dense and block diagonal sums and differences preserve operand order")
{
	Hoppy::BlockDiagonalMatrix<double> block{1, 2};
	fill(block);
	Eigen::Matrix3d dense;
	dense.setRandom();
	const Eigen::Matrix3d blockDense = block.toDense();

	const auto densePlusBlock = dense + block;
	const auto blockPlusDense = block + dense;
	const auto denseMinusBlock = dense - block;
	const auto blockMinusDense = block - dense;
	static_assert(std::is_same_v<typename decltype(densePlusBlock)::Scalar, double>);
	Hoppy::Test::requireApprox(densePlusBlock.eval(), dense + blockDense);
	Hoppy::Test::requireApprox(blockPlusDense.eval(), blockDense + dense);
	Hoppy::Test::requireApprox(denseMinusBlock.eval(), dense - blockDense);
	Hoppy::Test::requireApprox(blockMinusDense.eval(), blockDense - dense);
}

TEST_CASE("dense interop supports row-major mapped strided and lazy dense operands")
{
	Hoppy::BlockDiagonalMatrix<double> block{1, 2};
	fill(block);
	Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rowMajor;
	rowMajor.setRandom();
	Hoppy::Test::requireApprox((rowMajor + block).eval(), rowMajor + block.toDense());

	Eigen::Matrix<double, 6, 3> storage;
	storage.setRandom();
	Eigen::Map<const Eigen::Matrix3d, 0, Eigen::Stride<6, 1>> strided(
	        storage.data(), Eigen::Stride<6, 1>{});
	Hoppy::Test::requireApprox((block - strided).eval(), block.toDense() - strided);

	Eigen::Matrix3d other;
	other.setRandom();
	const auto lazy = rowMajor + other;
	Hoppy::Test::requireApprox((lazy + block).eval(), lazy.eval() + block.toDense());
	Hoppy::Test::requireApprox((2.0 * (block - lazy)).eval(),
	                           2.0 * (block.toDense() - lazy.eval()));
}

TEST_CASE("dense interop supports destination aliasing on the dense side")
{
	Hoppy::BlockDiagonalMatrix<double> block{1, 2};
	fill(block);
	Eigen::Matrix3d destination;
	destination.setRandom();
	const Eigen::Matrix3d original = destination;
	destination = destination + block;
	Hoppy::Test::requireApprox(destination, original + block.toDense());
	destination = destination - block;
	Hoppy::Test::requireApprox(destination, original);
}

TEST_CASE("dense interop owns block-expression temporaries according to Eigen nesting rules")
{
	Hoppy::BlockDiagonalMatrix<double> a{1, 2};
	Hoppy::BlockDiagonalMatrix<double> b{1, 2};
	fill(a);
	b.setConstant(2.0);
	Eigen::Matrix3d dense;
	dense.setRandom();

	const auto sum = dense + (a.transpose() / 2.0);
	Hoppy::Test::requireApprox((sum + dense).eval(),
	                           dense + a.toDense().transpose() / 2.0 + dense);
	const auto difference = (a + b) - dense;
	Hoppy::Test::requireApprox((difference - dense).eval(),
	                           a.toDense() + b.toDense() - dense - dense);
}

TEST_CASE("dense interop handles empty matrices and Eigen scalar promotion")
{
	Hoppy::BlockDiagonalMatrix<double> empty;
	Eigen::MatrixXd dense(0, 0);
	REQUIRE((dense + empty).eval().rows() == 0);
	REQUIRE((empty - dense).eval().cols() == 0);

	using Complex = std::complex<double>;
	Hoppy::BlockDiagonalMatrix<double> block{1, 2};
	fill(block);
	Eigen::MatrixXcd complex = Eigen::MatrixXcd::Constant(3, 3, Complex{1.0, 2.0});
	const auto result = complex + block;
	static_assert(std::is_same_v<typename decltype(result)::Scalar, Complex>);
	Hoppy::Test::requireApprox(result.eval(), complex + block.toDense());

	using InvalidDense = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic>;
	using InvalidBlock = Hoppy::BlockDiagonalMatrix<int>;
	static_assert(!has_sum<InvalidDense, InvalidBlock>::value);
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("dense interop enforces dimensions and destination storage separation")
{
	Hoppy::BlockDiagonalMatrix<double> block{1, 2};
	Eigen::MatrixXd wrong(2, 2);
	REQUIRE_THROWS_AS((void)(wrong + block), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS((void)(block - wrong), Hoppy::Test::EigenAssertionFailure);

	Hoppy::BlockDiagonalMatrix<double> one{1};
	Eigen::Map<Eigen::Matrix<double, 1, 1>> overlapping(one.data());
	REQUIRE_THROWS_AS(overlapping = overlapping + one, Hoppy::Test::EigenAssertionFailure);
}
#endif
