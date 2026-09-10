// SPDX-License-Identifier: MIT

// Triangular-compressed specification: D5-D10, D14, D18; sections 3-6, 9, and 15-16.

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/TriangularCompressedMatrix.hpp>

#include <Eigen/Core>

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

struct HoppyThrowingScalar
{
	int value = 0;
	static bool throwOnAssignment;

	HoppyThrowingScalar() = default;
	HoppyThrowingScalar(const int input) : value(input) {}
	HoppyThrowingScalar(const HoppyThrowingScalar&) = default;
	HoppyThrowingScalar& operator=(const HoppyThrowingScalar& other)
	{
		if (throwOnAssignment) throw std::runtime_error("scalar assignment failure");
		value = other.value;
		return *this;
	}
	friend bool operator==(const HoppyThrowingScalar& left, const HoppyThrowingScalar& right)
	{
		return left.value == right.value;
	}
};

bool HoppyThrowingScalar::throwOnAssignment = false;

namespace Eigen
{
	template <>
	struct NumTraits<HoppyThrowingScalar> : GenericNumTraits<HoppyThrowingScalar>
	{
		using Real = HoppyThrowingScalar;
		using NonInteger = HoppyThrowingScalar;
		using Nested = HoppyThrowingScalar;
		enum
		{
			IsComplex = 0
		};
	};
}  // namespace Eigen

namespace
{
	template <typename T, typename = void>
	struct has_inner_stride : std::false_type
	{};
	template <typename T>
	struct has_inner_stride<T, std::void_t<decltype(std::declval<const T&>().innerStride())>>
	    : std::true_type
	{};

	template <typename T, typename = void>
	struct has_one_index_call : std::false_type
	{};
	template <typename T>
	struct has_one_index_call<T, std::void_t<decltype(std::declval<T&>()(Eigen::Index{}))>>
	    : std::true_type
	{};

	static_assert(!has_inner_stride<Hoppy::SymmetricMatrixXd>::value);
	static_assert((Eigen::internal::traits<Hoppy::SymmetricMatrixXd>::Flags & Eigen::DirectAccessBit) == 0);
	static_assert(has_one_index_call<Hoppy::SymmetricMatrix<double, 1>>::value);
	static_assert(!has_one_index_call<Hoppy::SymmetricMatrix<double, 2>>::value);
	static_assert(!has_one_index_call<Hoppy::SymmetricMatrixXd>::value);
	static_assert(std::is_same_v<decltype(std::declval<const Hoppy::SymmetricMatrixXd&>().coeff(0, 0)),
	                             double>);
	static_assert(std::is_convertible_v<decltype(std::declval<Hoppy::SymmetricMatrixXd&>().coeffRef(0, 0)),
	                                    double>);
	using SymmetricPolicy = Hoppy::Detail::TriangularCompressedCoefficientPolicy<
	        double, Hoppy::Detail::SymmetricTag, Hoppy::TrianglePacking::Lower>;
	static_assert(noexcept(SymmetricPolicy::isValidIndex(3, 0, 0)));

	template <typename Matrix>
	void requireShape(const Matrix& matrix, Eigen::Index dimension)
	{
		REQUIRE(matrix.rows() == dimension);
		REQUIRE(matrix.cols() == dimension);
		REQUIRE(matrix.dimension() == dimension);
		REQUIRE(matrix.size() == dimension * dimension);
		REQUIRE(matrix.storedSize() == dimension * (dimension + 1) / 2);
	}

	template <typename... Matrices>
	struct matrix_list
	{};

	template <typename Matrix>
	void requireDynamicDimensions()
	{
		requireShape(Matrix(0), 0);
		requireShape(Matrix(1), 1);
		requireShape(Matrix(4), 4);
	}

	template <typename... Matrices>
	void requireAllDynamicDimensions(matrix_list<Matrices...>)
	{
		(requireDynamicDimensions<Matrices>(), ...);
	}

	template <typename... Matrices>
	void requireAllFixedDimensions(matrix_list<Matrices...>)
	{
		(requireShape(Matrices{}, 3), ...);
	}

	template <typename Matrix>
	void initializeTwoSided(Matrix& matrix)
	{
		using Scalar = typename Matrix::Scalar;
		for (Eigen::Index row = 0; row < matrix.dimension(); ++row)
			for (Eigen::Index column = 0; column < matrix.dimension(); ++column)
				if ((Matrix::PackingValue == Hoppy::TrianglePacking::Lower && row >= column)
				    || (Matrix::PackingValue == Hoppy::TrianglePacking::Upper && row <= column))
					matrix(row, column) = Scalar(10 * row + column + 1);
	}

	template <typename Matrix>
	void requireSymmetricFamily()
	{
		Matrix matrix(4);
		initializeTwoSided(matrix);
		for (Eigen::Index row = 0; row < 4; ++row)
			for (Eigen::Index column = 0; column < 4; ++column)
			{
				REQUIRE(matrix.coeff(row, column) == matrix.coeff(column, row));
				if ((Matrix::PackingValue == Hoppy::TrianglePacking::Lower && row >= column)
				    || (Matrix::PackingValue == Hoppy::TrianglePacking::Upper && row <= column))
					REQUIRE(matrix.data()[Hoppy::Detail::checkedPackedOffset(
					                row, column, 4, Matrix::PackingValue)] == matrix.coeff(row, column));
			}
	}

	template <typename Matrix>
	void requireAntiSymmetricFamily()
	{
		Matrix matrix(4);
		for (Eigen::Index row = 0; row < 4; ++row)
			for (Eigen::Index column = 0; column < 4; ++column)
				if (row != column
				    && ((Matrix::PackingValue == Hoppy::TrianglePacking::Lower && row > column)
				        || (Matrix::PackingValue == Hoppy::TrianglePacking::Upper && row < column)))
					matrix(row, column) = typename Matrix::Scalar(10 * row + column + 1);
		for (Eigen::Index row = 0; row < 4; ++row)
			for (Eigen::Index column = 0; column < 4; ++column)
			{
				REQUIRE(matrix.coeff(row, column) == -matrix.coeff(column, row));
				if ((Matrix::PackingValue == Hoppy::TrianglePacking::Lower && row >= column)
				    || (Matrix::PackingValue == Hoppy::TrianglePacking::Upper && row <= column))
					REQUIRE(matrix.data()[Hoppy::Detail::checkedPackedOffset(
					                row, column, 4, Matrix::PackingValue)] == matrix.coeff(row, column));
			}
	}

	template <typename Matrix>
	void requireTriangularFamily(bool upper)
	{
		Matrix matrix(4);
		for (Eigen::Index row = 0; row < 4; ++row)
			for (Eigen::Index column = 0; column < 4; ++column)
				if (upper ? row <= column : row >= column)
					matrix(row, column) = typename Matrix::Scalar(10 * row + column + 1);
		for (Eigen::Index row = 0; row < 4; ++row)
			for (Eigen::Index column = 0; column < 4; ++column)
				if (upper ? row > column : row < column)
					REQUIRE(matrix.coeff(row, column) == 0.0);
				else
					REQUIRE(matrix.data()[Hoppy::Detail::checkedPackedOffset(
					                row, column, 4, Matrix::PackingValue)] == matrix.coeff(row, column));
	}
}  // namespace

TEST_CASE("owning shapes cover empty, scalar, dynamic, and fixed matrices")
{
	REQUIRE(SymmetricPolicy::isValidIndex(3, 2, 2));
	REQUIRE_FALSE(SymmetricPolicy::isValidIndex(3, 3, 0));
	REQUIRE(Hoppy::SymmetricMatrixXd::requiredStoredSize(4) == 10);
	REQUIRE(Hoppy::SymmetricMatrix<double, 3>::requiredStoredSize(3) == 6);
	Hoppy::SymmetricMatrixXd empty;
	requireShape(empty, 0);
	Hoppy::SymmetricMatrix<double, 1> scalar(1, 1);
	requireShape(scalar, 1);
	scalar(0) = 3.0;
	REQUIRE(scalar.coeff(0, 0) == 3.0);
	Hoppy::SymmetricMatrix<double, 3> fixed;
	requireShape(fixed, 3);
	Hoppy::SymmetricMatrixXd dynamic(4);
	requireShape(dynamic, 4);
	static_assert(std::is_same_v<decltype(std::as_const(dynamic).data()), const double*>);
	static_assert(std::is_same_v<decltype(dynamic.data()), double*>);
}

TEST_CASE("all distinct families and packing sides support required dynamic and fixed dimensions")
{
	using namespace Hoppy;
	using Complex = std::complex<double>;
	requireAllDynamicDimensions(matrix_list<
	        UpperTriangularMatrix<double>, UpperTriangularMatrix<double, Eigen::Dynamic, TrianglePacking::Upper>,
	        LowerTriangularMatrix<double>, LowerTriangularMatrix<double, Eigen::Dynamic, TrianglePacking::Upper>,
	        SymmetricMatrix<double>, SymmetricMatrix<double, Eigen::Dynamic, TrianglePacking::Upper>,
	        AntiSymmetricMatrix<double>, AntiSymmetricMatrix<double, Eigen::Dynamic, TrianglePacking::Upper>,
	        UpperTriangularMatrix<Complex>, UpperTriangularMatrix<Complex, Eigen::Dynamic, TrianglePacking::Upper>,
	        LowerTriangularMatrix<Complex>, LowerTriangularMatrix<Complex, Eigen::Dynamic, TrianglePacking::Upper>,
	        SymmetricMatrix<Complex>, SymmetricMatrix<Complex, Eigen::Dynamic, TrianglePacking::Upper>,
	        AntiSymmetricMatrix<Complex>, AntiSymmetricMatrix<Complex, Eigen::Dynamic, TrianglePacking::Upper>,
	        HermitianMatrix<Complex>, HermitianMatrix<Complex, Eigen::Dynamic, TrianglePacking::Upper>,
	        AntiHermitianMatrix<Complex>, AntiHermitianMatrix<Complex, Eigen::Dynamic, TrianglePacking::Upper>>{});
	requireAllFixedDimensions(matrix_list<
	        UpperTriangularMatrix<double, 3>, UpperTriangularMatrix<double, 3, TrianglePacking::Upper>,
	        LowerTriangularMatrix<double, 3>, LowerTriangularMatrix<double, 3, TrianglePacking::Upper>,
	        SymmetricMatrix<double, 3>, SymmetricMatrix<double, 3, TrianglePacking::Upper>,
	        AntiSymmetricMatrix<double, 3>, AntiSymmetricMatrix<double, 3, TrianglePacking::Upper>,
	        UpperTriangularMatrix<Complex, 3>, UpperTriangularMatrix<Complex, 3, TrianglePacking::Upper>,
	        LowerTriangularMatrix<Complex, 3>, LowerTriangularMatrix<Complex, 3, TrianglePacking::Upper>,
	        SymmetricMatrix<Complex, 3>, SymmetricMatrix<Complex, 3, TrianglePacking::Upper>,
	        AntiSymmetricMatrix<Complex, 3>, AntiSymmetricMatrix<Complex, 3, TrianglePacking::Upper>,
	        HermitianMatrix<Complex, 3>, HermitianMatrix<Complex, 3, TrianglePacking::Upper>,
	        AntiHermitianMatrix<Complex, 3>, AntiHermitianMatrix<Complex, 3, TrianglePacking::Upper>>{});
}

TEST_CASE("all real matrix families obey their logical identities in either packing")
{
	requireTriangularFamily<Hoppy::UpperTriangularMatrix<double, Eigen::Dynamic,
	                                                    Hoppy::TrianglePacking::Lower>>(true);
	requireTriangularFamily<Hoppy::UpperTriangularMatrix<double, Eigen::Dynamic,
	                                                    Hoppy::TrianglePacking::Upper>>(true);
	requireTriangularFamily<Hoppy::LowerTriangularMatrix<double, Eigen::Dynamic,
	                                                    Hoppy::TrianglePacking::Lower>>(false);
	requireTriangularFamily<Hoppy::LowerTriangularMatrix<double, Eigen::Dynamic,
	                                                    Hoppy::TrianglePacking::Upper>>(false);
	requireSymmetricFamily<Hoppy::SymmetricMatrix<double, Eigen::Dynamic,
	                                               Hoppy::TrianglePacking::Lower>>();
	requireSymmetricFamily<Hoppy::SymmetricMatrix<double, Eigen::Dynamic,
	                                               Hoppy::TrianglePacking::Upper>>();
	requireAntiSymmetricFamily<Hoppy::AntiSymmetricMatrix<double, Eigen::Dynamic,
	                                                       Hoppy::TrianglePacking::Lower>>();
	requireAntiSymmetricFamily<Hoppy::AntiSymmetricMatrix<double, Eigen::Dynamic,
	                                                       Hoppy::TrianglePacking::Upper>>();
}

TEST_CASE("complex triangular symmetric and anti-symmetric families preserve their identities")
{
	using Complex = std::complex<double>;
	requireTriangularFamily<Hoppy::UpperTriangularMatrix<Complex, Eigen::Dynamic,
	                                                    Hoppy::TrianglePacking::Lower>>(true);
	requireTriangularFamily<Hoppy::LowerTriangularMatrix<Complex, Eigen::Dynamic,
	                                                    Hoppy::TrianglePacking::Upper>>(false);
	requireSymmetricFamily<Hoppy::SymmetricMatrix<Complex, Eigen::Dynamic,
	                                               Hoppy::TrianglePacking::Lower>>();
	requireSymmetricFamily<Hoppy::SymmetricMatrix<Complex, Eigen::Dynamic,
	                                               Hoppy::TrianglePacking::Upper>>();
	requireAntiSymmetricFamily<Hoppy::AntiSymmetricMatrix<Complex, Eigen::Dynamic,
	                                                       Hoppy::TrianglePacking::Lower>>();
	requireAntiSymmetricFamily<Hoppy::AntiSymmetricMatrix<Complex, Eigen::Dynamic,
	                                                       Hoppy::TrianglePacking::Upper>>();
}

TEST_CASE("complex Hermitian families conjugate and negate reflected coefficients")
{
	using Complex = std::complex<double>;
	for (const auto packing : {Hoppy::TrianglePacking::Lower, Hoppy::TrianglePacking::Upper})
	{
		if (packing == Hoppy::TrianglePacking::Lower)
		{
			Hoppy::HermitianMatrix<Complex> hermitian(4);
			hermitian(2, 0) = Complex(3, 4);
			hermitian(1, 1) = Complex(5, 0);
			REQUIRE(hermitian.coeff(0, 2) == Complex(3, -4));
			Hoppy::AntiHermitianMatrix<Complex> anti(4);
			anti(2, 0) = Complex(3, 4);
			anti(1, 1) = Complex(0, 5);
			REQUIRE(anti.coeff(0, 2) == Complex(-3, 4));
		}
		else
		{
			Hoppy::HermitianMatrix<Complex, Eigen::Dynamic, Hoppy::TrianglePacking::Upper> hermitian(4);
			hermitian(0, 2) = Complex(3, 4);
			REQUIRE(hermitian.coeff(2, 0) == Complex(3, -4));
			Hoppy::AntiHermitianMatrix<Complex, Eigen::Dynamic, Hoppy::TrianglePacking::Upper> anti(4);
			anti(0, 2) = Complex(3, 4);
			REQUIRE(anti.coeff(2, 0) == Complex(-3, 4));
		}
	}
}

TEST_CASE("reflected Hermitian writes transform values before packed storage")
{
	using Complex = std::complex<double>;
	Hoppy::HermitianMatrix<Complex> hermitian(3);
	hermitian(0, 2) = Complex(3, 4);
	REQUIRE(hermitian.coeff(0, 2) == Complex(3, 4));
	REQUIRE(hermitian.coeff(2, 0) == Complex(3, -4));
	REQUIRE(hermitian.data()[3] == Complex(3, -4));

	Hoppy::AntiHermitianMatrix<Complex> antiHermitian(3);
	antiHermitian(0, 2) = Complex(3, 4);
	REQUIRE(antiHermitian.coeff(0, 2) == Complex(3, 4));
	REQUIRE(antiHermitian.coeff(2, 0) == Complex(-3, 4));
	REQUIRE(antiHermitian.data()[3] == Complex(-3, 4));
}

TEST_CASE("packed order is exact and does not alter logical symmetric values")
{
	Hoppy::SymmetricMatrixXd lower(3);
	lower(0, 0) = 1; lower(1, 0) = 2; lower(1, 1) = 3;
	lower(2, 0) = 4; lower(2, 1) = 5; lower(2, 2) = 6;
	Hoppy::SymmetricMatrix<double, Eigen::Dynamic, Hoppy::TrianglePacking::Upper> upper(3);
	upper(0, 0) = 1; upper(0, 1) = 2; upper(1, 1) = 3;
	upper(0, 2) = 4; upper(1, 2) = 5; upper(2, 2) = 6;
	for (Eigen::Index index = 0; index < 6; ++index)
	{
		REQUIRE(lower.data()[index] == index + 1);
		REQUIRE(upper.data()[index] == index + 1);
	}
	for (Eigen::Index row = 0; row < 3; ++row)
		for (Eigen::Index column = 0; column < 3; ++column)
			REQUIRE(lower.coeff(row, column) == upper.coeff(row, column));
}

TEST_CASE("coefficient proxies support direct reflected compound and proxy assignments")
{
	Hoppy::SymmetricMatrixXd matrix(3);
	matrix(2, 0) = 4.0;
	matrix(0, 2) += 2.0;
	matrix(2, 0) -= 1.0;
	matrix(0, 2) *= 3.0;
	matrix(2, 0) /= 5.0;
	REQUIRE(matrix.coeff(0, 2) == 3.0);
	matrix(1, 0) = matrix(0, 2);
	REQUIRE(matrix.coeff(0, 1) == 3.0);
	matrix(0, 2) = matrix(2, 0);
	REQUIRE(matrix.coeff(0, 2) == 3.0);

	Hoppy::AntiSymmetricMatrixXd anti(3);
	anti(2, 0) = 7.0;
	anti(0, 2) += 2.0;
	REQUIRE(anti.coeff(0, 2) == -5.0);
	REQUIRE(anti.coeff(2, 0) == 5.0);
	anti(0, 2) = anti(2, 0);
	REQUIRE(anti.coeff(0, 2) == 5.0);
	REQUIRE(anti.coeff(2, 0) == -5.0);
}

TEST_CASE("copy move swap and conservative resize follow value-container semantics")
{
	Hoppy::SymmetricMatrixXd original(3);
	initializeTwoSided(original);
	auto copy = original;
	copy(2, 0) = 99.0;
	REQUIRE(original.coeff(2, 0) != copy.coeff(2, 0));
	auto* self = &original;
	original = *self;
	REQUIRE(original.coeff(2, 1) == 22.0);

	auto moved = std::move(copy);
	REQUIRE(moved.coeff(2, 0) == 99.0);
	swap(original, moved);
	REQUIRE(original.coeff(2, 0) == 99.0);

	original.conservativeResize(5);
	REQUIRE(original.coeff(2, 0) == 99.0);
	original.conservativeResize(2, 2);
	REQUIRE(original.coeff(1, 0) == 11.0);
	original.resizeLike(Eigen::MatrixXd(4, 4));
	requireShape(original, 4);

	Hoppy::SymmetricMatrix<double, Eigen::Dynamic, Hoppy::TrianglePacking::Upper> upper(3);
	initializeTwoSided(upper);
	upper.conservativeResize(5);
	REQUIRE(upper.coeff(0, 2) == 3.0);
	upper.conservativeResize(2);
	REQUIRE(upper.coeff(0, 1) == 2.0);
}

TEST_CASE("aligned and unaligned owning buffers honor their allocator contracts")
{
	Hoppy::SymmetricMatrix<double, Eigen::Dynamic, Hoppy::TrianglePacking::Lower, 0> aligned(4);
	Hoppy::SymmetricMatrix<double, Eigen::Dynamic, Hoppy::TrianglePacking::Lower,
	                       Eigen::DontAlign> unaligned(4);
	REQUIRE(reinterpret_cast<std::uintptr_t>(aligned.data()) % EIGEN_MAX_ALIGN_BYTES == 0);
	REQUIRE(unaligned.data() != nullptr);
}

TEST_CASE("failed conservative resize leaves the original object unchanged")
{
	Hoppy::SymmetricMatrix<HoppyThrowingScalar> matrix(2);
	matrix(0, 0) = HoppyThrowingScalar(1);
	matrix(1, 0) = HoppyThrowingScalar(2);
	matrix(1, 1) = HoppyThrowingScalar(3);
	HoppyThrowingScalar::throwOnAssignment = true;
	REQUIRE_THROWS_AS(matrix.conservativeResize(3), std::runtime_error);
	HoppyThrowingScalar::throwOnAssignment = false;
	REQUIRE(matrix.dimension() == 2);
	REQUIRE(matrix.coeff(0, 0).value == 1);
	REQUIRE(matrix.coeff(1, 0).value == 2);
	REQUIRE(matrix.coeff(1, 1).value == 3);
}

#ifdef HOPPY_TEST_EIGEN_ASSERT_THROWS
TEST_CASE("invalid coefficient and diagonal writes assert before changing storage")
{
	using FixedSymmetric = Hoppy::SymmetricMatrix<double, 3>;
	REQUIRE_THROWS_AS(FixedSymmetric::requiredStoredSize(4),
	                  Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(Hoppy::SymmetricMatrixXd::requiredStoredSize(-1),
	                  Hoppy::Test::EigenAssertionFailure);
	Hoppy::UpperTriangularMatrixXd upper(3);
	upper(0, 1) = 7.0;
	REQUIRE_THROWS_AS(upper(1, 0) = 3.0, Hoppy::Test::EigenAssertionFailure);
	REQUIRE(upper.coeff(0, 1) == 7.0);
	REQUIRE_THROWS_AS(upper.coeff(-1, 0), Hoppy::Test::EigenAssertionFailure);

	Hoppy::AntiSymmetricMatrixXd anti(3);
	REQUIRE_THROWS_AS(anti(1, 1) = 1.0, Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(anti(1, 1) += 1.0, Hoppy::Test::EigenAssertionFailure);
	REQUIRE(anti.coeff(1, 1) == 0.0);

	using Complex = std::complex<double>;
	Hoppy::HermitianMatrix<Complex> hermitian(3);
	REQUIRE_THROWS_AS(hermitian(1, 1) = Complex(1, 2), Hoppy::Test::EigenAssertionFailure);
	REQUIRE(hermitian.coeff(1, 1) == Complex(0, 0));
	Hoppy::AntiHermitianMatrix<Complex> antiHermitian(3);
	REQUIRE_THROWS_AS(antiHermitian(1, 1) = Complex(1, 2), Hoppy::Test::EigenAssertionFailure);
	REQUIRE(antiHermitian.coeff(1, 1) == Complex(0, 0));

	Hoppy::SymmetricMatrix<double, 3> fixed;
	REQUIRE_THROWS_AS(fixed.resize(4), Hoppy::Test::EigenAssertionFailure);
	REQUIRE(fixed.dimension() == 3);
	REQUIRE_THROWS_AS(fixed.resize(3, 2), Hoppy::Test::EigenAssertionFailure);
}
#endif
