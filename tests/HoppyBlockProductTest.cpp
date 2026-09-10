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

	struct UnsupportedScalar {};

	template <typename Matrix>
	void fill(Matrix& matrix, const typename Matrix::Scalar offset = typename Matrix::Scalar{})
	{
		for (Eigen::Index index = 0; index < matrix.storedSize(); ++index)
			matrix.data()[index] = static_cast<typename Matrix::Scalar>(index + 1) + offset;
	}
}

TEST_CASE("blockwise products are lazy and match dense multiplication")
{
	Hoppy::BlockDiagonalMatrix<double> lhs{1, 2, 3};
	Hoppy::BlockDiagonalMatrix<double> rhs{1, 2, 3};
	fill(lhs);
	fill(rhs, 2.0);
	const auto lhsDense = lhs.toDense();
	const auto rhsDense = rhs.toDense();
	const auto product = lhs * rhs;

	using ExpectedScalar = typename Eigen::ScalarBinaryOpTraits<
	        double, double, Eigen::internal::scalar_product_op<double, double>>::ReturnType;
	static_assert(std::is_same_v<typename Eigen::internal::traits<decltype(product)>::Scalar,
	                             ExpectedScalar>);
	static_assert(std::is_same_v<std::remove_cv_t<decltype(product)>,
	                             Hoppy::Detail::BlockwiseProduct<decltype(lhs), decltype(rhs)>>);
	Hoppy::Test::requireApprox(product.toDense(), lhsDense * rhsDense);

	lhs[2](0, 0) = 100.0;
	Hoppy::Test::requireApprox(product[0], lhs[0] * rhs[0]);
	Hoppy::Test::requireApprox(product[2], lhs[2] * rhs[2]);
}

TEST_CASE("complex and permitted mixed-scalar products preserve Eigen promotion")
{
	using Complex = std::complex<double>;
	Hoppy::BlockDiagonalMatrix<double> real{1, 2};
	Hoppy::BlockDiagonalMatrix<Complex> complex{1, 2};
	fill(real);
	fill(complex, Complex{1.0, 2.0});
	const auto product = real * complex;
	using DenseProduct = decltype(std::declval<const Eigen::MatrixXd&>()
	                            * std::declval<const Eigen::MatrixXcd&>());
	static_assert(std::is_same_v<typename Eigen::internal::traits<decltype(product)>::Scalar,
	                             typename DenseProduct::Scalar>);
	Hoppy::Test::requireApprox(product.toDense(), real.toDense() * complex.toDense());
	static_assert(!has_product<Hoppy::BlockDiagonalMatrix<float>,
	                           Hoppy::BlockDiagonalMatrix<double>>::value);
	static_assert(!has_product<Hoppy::BlockDiagonalMatrix<double>, UnsupportedScalar>::value);
}

TEST_CASE("product assignments are block-local and alias safe")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	fill(matrix);
	const auto dense = matrix.toDense();
	matrix = matrix * matrix;
	Hoppy::Test::requireApprox(matrix.toDense(), dense * dense);

	Hoppy::BlockDiagonalMatrix<double> compound{1, 2};
	fill(compound);
	compound *= compound;
	Hoppy::Test::requireApprox(compound.toDense(), dense * dense);
}

TEST_CASE("product assignment with different blocking evaluates before replacement")
{
	Hoppy::BlockDiagonalMatrix<double> lhs{1, 2};
	Hoppy::BlockDiagonalMatrix<double> rhs{1, 2};
	fill(lhs);
	fill(rhs, 1.0);
	const auto expected = (lhs.toDense() * rhs.toDense()).eval();

	Hoppy::BlockDiagonalMatrix<double> destination{3};
	destination.setConstant(-1.0);
	destination = lhs * rhs;
	REQUIRE(destination.blockingInfo() == std::vector<Eigen::Index>{1, 2});
	Hoppy::Test::requireApprox(destination.toDense(), expected);
}

TEST_CASE("product expression prvalues remain chainable in either operand position")
{
	Hoppy::BlockDiagonalMatrix<double> a{1, 2};
	Hoppy::BlockDiagonalMatrix<double> b{1, 2};
	Hoppy::BlockDiagonalMatrix<double> c{1, 2};
	fill(a);
	fill(b, 1.0);
	fill(c, 2.0);
	const auto ad = a.toDense();
	const auto bd = b.toDense();
	const auto cd = c.toDense();

	Hoppy::Test::requireApprox(((a * b) * c).toDense(), (ad * bd) * cd);
	Hoppy::Test::requireApprox((a * (b * c)).toDense(), ad * (bd * cd));
	Hoppy::Test::requireApprox(((a * b).transpose() / 2.0).toDense(),
	                           (ad * bd).transpose() / 2.0);
	Hoppy::Test::requireApprox(((a + b) * (c - a)).toDense(), (ad + bd) * (cd - ad));
}

TEST_CASE("asDiagonal retains Eigen diagonal blocks in products")
{
	Hoppy::BlockDiagonalMatrix<double> matrix{1, 2};
	Hoppy::BlockVector<double> diagonal{1, 2};
	fill(matrix);
	diagonal.asDense() << 2.0, 3.0, 4.0;
	const auto result = matrix * diagonal.asDiagonal();
	Hoppy::Test::requireApprox(result.toDense(),
	                           matrix.toDense() * diagonal.asDense().asDiagonal().toDenseMatrix());
}

#ifdef HOPPY_TEST_EIGEN_ASSERT_THROWS
TEST_CASE("products and product compounds require equal blocking")
{
	Hoppy::BlockDiagonalMatrix<double> lhs{1, 2};
	Hoppy::BlockDiagonalMatrix<double> rhs{3};
	REQUIRE_THROWS_AS((void)(lhs * rhs), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(lhs *= rhs, Hoppy::Test::EigenAssertionFailure);
}
#endif
