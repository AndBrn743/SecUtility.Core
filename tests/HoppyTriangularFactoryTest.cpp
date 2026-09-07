// SPDX-License-Identifier: MIT

// Triangular-compressed specification: D11-D14, D20; section 7.

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/TriangularCompressedMatrix.hpp>

#include <Eigen/Core>

#include <array>
#include <complex>
#include <type_traits>
#include <utility>

namespace
{
	template <typename T, typename = void> struct has_ones : std::false_type {};
	template <typename T> struct has_ones<T, std::void_t<decltype(T::Ones())>> : std::true_type {};
	template <typename T, typename = void> struct has_identity : std::false_type {};
	template <typename T> struct has_identity<T, std::void_t<decltype(T::Identity())>> : std::true_type {};
	template <typename T, typename = void> struct has_set_ones : std::false_type {};
	template <typename T>
	struct has_set_ones<T, std::void_t<decltype(std::declval<T&>().setOnes())>> : std::true_type {};
	template <typename T, typename E, typename = void> struct has_from_upper : std::false_type {};
	template <typename T, typename E>
	struct has_from_upper<T, E, std::void_t<decltype(T::FromUpper(std::declval<const E&>()))>>
	    : std::true_type {};
	template <typename T, typename E, typename = void> struct has_from_lower : std::false_type {};
	template <typename T, typename E>
	struct has_from_lower<T, E, std::void_t<decltype(T::FromLower(std::declval<const E&>()))>>
	    : std::true_type {};

	using Dense = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
	static_assert(has_ones<Hoppy::SymmetricMatrixXd>::value);
	static_assert(has_identity<Hoppy::HermitianMatrixXcd>::value);
	static_assert(!has_ones<Hoppy::AntiSymmetricMatrixXd>::value);
	static_assert(!has_identity<Hoppy::AntiHermitianMatrixXcd>::value);
	static_assert(has_set_ones<Hoppy::HermitianMatrixXcd>::value);
	static_assert(!has_set_ones<Hoppy::AntiSymmetricMatrixXd>::value);
	static_assert(has_from_upper<Hoppy::UpperTriangularMatrixXd, Dense>::value);
	static_assert(!has_from_lower<Hoppy::UpperTriangularMatrixXd, Dense>::value);
	static_assert(has_from_lower<Hoppy::LowerTriangularMatrixXd, Dense>::value);
	static_assert(!has_from_upper<Hoppy::LowerTriangularMatrixXd, Dense>::value);

	struct SelectedTriangleExpression
	{
		Eigen::Index rows() const { return 3; }
		Eigen::Index cols() const { return 3; }
		double coeff(Eigen::Index row, Eigen::Index column) const
		{
			REQUIRE(row <= column);
			++reads;
			return static_cast<double>(10 * row + column + 1);
		}
		mutable int reads = 0;
	};
}

TEST_CASE("Eigen-style fills preserve each family invariant")
{
	auto symmetric = Hoppy::SymmetricMatrixXd::Constant(3, 4.0);
	REQUIRE(symmetric.coeff(0, 2) == 4.0);
	REQUIRE(symmetric.coeff(2, 0) == 4.0);
	symmetric.setIdentity();
	REQUIRE(symmetric.coeff(0, 0) == 1.0);
	REQUIRE(symmetric.coeff(0, 1) == 0.0);

	auto anti = Hoppy::AntiSymmetricMatrixXd::Random(4);
	for (Eigen::Index index = 0; index < anti.dimension(); ++index)
		REQUIRE(anti.coeff(index, index) == 0.0);

	auto hermitian = Hoppy::HermitianMatrixXcd::Random(4);
	auto antiHermitian = Hoppy::AntiHermitianMatrixXcd::Random(4);
	for (Eigen::Index index = 0; index < 4; ++index)
	{
		REQUIRE(hermitian.coeff(index, index).imag() == 0.0);
		REQUIRE(antiHermitian.coeff(index, index).real() == 0.0);
	}
	REQUIRE(Hoppy::UpperTriangularMatrix<double, 3>::Zero().dimension() == 3);
	REQUIRE(Hoppy::SymmetricMatrixXd::Zero().dimension() == 0);
}

TEST_CASE("setters are shared by owning matrices and mutable maps")
{
	std::array<double, 6> storage{};
	Eigen::Map<Hoppy::SymmetricMatrix<double, 3>> map(storage.data());
	REQUIRE(&map.setOnes() == &map);
	for (double value : storage) REQUIRE(value == 1.0);
	map.setIdentity();
	REQUIRE(map.coeff(2, 2) == 1.0);
	REQUIRE(map.coeff(2, 0) == 0.0);
}

#ifndef EIGEN_NO_DEBUG
TEST_CASE("invalid constant is transactional")
{
	using Complex = std::complex<double>;
	auto matrix = Hoppy::HermitianMatrixXcd::Constant(3, Complex(2, 0));
	REQUIRE_THROWS_AS(matrix.setConstant(Complex(7, 1)), Hoppy::Test::EigenAssertionFailure);
	for (Eigen::Index index = 0; index < matrix.storedSize(); ++index)
		REQUIRE(matrix.data()[index] == Complex(2, 0));
}
#endif

TEST_CASE("selected factories read only their named triangle")
{
	SelectedTriangleExpression expression;
	auto symmetric = Hoppy::SymmetricMatrixXd::FromUpper(expression);
	REQUIRE(expression.reads == 6);
	REQUIRE(symmetric.coeff(2, 0) == 3.0);
	REQUIRE(symmetric.coeff(1, 2) == 13.0);
}

TEST_CASE("unchecked import selects its canonical side and canonicalizes the diagonal")
{
	using Complex = std::complex<double>;
	Eigen::Matrix<Complex, 2, 2> dense;
	dense << Complex(1, 9), Complex(2, 3), Complex(4, 5), Complex(6, 8);
	auto lower = Hoppy::HermitianMatrix<Complex, Eigen::Dynamic, Hoppy::TrianglePacking::Lower>
	                     ::FromUncheckedDense(dense);
	auto upper = Hoppy::HermitianMatrix<Complex, Eigen::Dynamic, Hoppy::TrianglePacking::Upper>
	                     ::FromUncheckedDense(dense);
	REQUIRE(lower.coeff(1, 0) == Complex(4, 5));
	REQUIRE(upper.coeff(0, 1) == Complex(2, 3));
	REQUIRE(lower.coeff(0, 0) == Complex(1, 0));
	REQUIRE(lower.coeff(1, 1) == Complex(6, 0));

	auto anti = Hoppy::AntiHermitianMatrixXcd::FromUncheckedDense(dense);
	REQUIRE(anti.coeff(0, 0) == Complex(0, 9));
}

TEST_CASE("modified dense import applies the normative projections")
{
	Eigen::Matrix<int, 2, 2> integers;
	integers << 1, 8, 2, 7;
	auto symmetric = Hoppy::SymmetricMatrix<int>::FromModifiedDense(integers);
	auto anti = Hoppy::AntiSymmetricMatrix<int>::FromModifiedDense(integers);
	REQUIRE(symmetric.coeff(0, 1) == 5);
	REQUIRE(anti.coeff(0, 1) == 3);
	REQUIRE(anti.coeff(1, 0) == -3);

	using Complex = std::complex<double>;
	Eigen::Matrix<Complex, 2, 2> dense;
	dense << Complex(1, 4), Complex(2, 6), Complex(8, 10), Complex(3, 12);
	auto hermitian = Hoppy::HermitianMatrixXcd::FromModifiedDense(dense);
	auto antiHermitian = Hoppy::AntiHermitianMatrixXcd::FromModifiedDense(dense);
	REQUIRE(hermitian.coeff(0, 1) == Complex(5, -2));
	REQUIRE(antiHermitian.coeff(0, 1) == Complex(-3, 8));
	REQUIRE(hermitian.coeff(0, 0) == Complex(1, 0));
	REQUIRE(antiHermitian.coeff(0, 0) == Complex(0, 4));
}
