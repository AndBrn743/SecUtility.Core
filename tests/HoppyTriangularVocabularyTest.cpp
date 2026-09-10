// SPDX-License-Identifier: MIT

// Triangular-compressed specification: D1-D10, D17-D18; sections 2-5.3 and 17.

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/TriangularCompressedMatrix.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>

struct HoppyCustomReal
{};

namespace Eigen
{
	template <>
	struct NumTraits<HoppyCustomReal> : GenericNumTraits<HoppyCustomReal>
	{
		using Real = HoppyCustomReal;
		using NonInteger = HoppyCustomReal;
		using Nested = HoppyCustomReal;
		enum
		{
			IsComplex = 0
		};
	};
}  // namespace Eigen

namespace
{
	using Hoppy::TrianglePacking;

	template <typename Matrix, typename Tag>
	inline constexpr bool has_structure_v = std::is_same_v<typename Matrix::StructureTag, Tag>;

	template <int Dimension, TrianglePacking Packing, int Options, typename = void>
	struct has_symmetric_matrix : std::false_type
	{};

	template <int Dimension, TrianglePacking Packing, int Options>
	struct has_symmetric_matrix<
	        Dimension, Packing, Options,
	        std::void_t<Hoppy::SymmetricMatrix<double, Dimension, Packing, Options>>>
	    : std::true_type
	{};

	template <typename Scalar, int Dimension, TrianglePacking Packing, int Options>
	constexpr bool vocabularyIsConsistent()
	{
		using Upper = Hoppy::UpperTriangularMatrix<Scalar, Dimension, Packing, Options>;
		using Lower = Hoppy::LowerTriangularMatrix<Scalar, Dimension, Packing, Options>;
		using Symmetric = Hoppy::SymmetricMatrix<Scalar, Dimension, Packing, Options>;
		using AntiSymmetric = Hoppy::AntiSymmetricMatrix<Scalar, Dimension, Packing, Options>;
		using Hermitian = Hoppy::HermitianMatrix<Scalar, Dimension, Packing, Options>;
		using AntiHermitian = Hoppy::AntiHermitianMatrix<Scalar, Dimension, Packing, Options>;
		return has_structure_v<Upper, Hoppy::Detail::UpperTriangularTag>
		       && has_structure_v<Lower, Hoppy::Detail::LowerTriangularTag>
		       && has_structure_v<Symmetric, Hoppy::Detail::SymmetricTag>
		       && has_structure_v<AntiSymmetric, Hoppy::Detail::AntiSymmetricTag>
		       && Upper::PackingValue == Packing && Upper::RowsAtCompileTime == Dimension
		       && (Eigen::NumTraits<Scalar>::IsComplex
		                   ? has_structure_v<Hermitian, Hoppy::Detail::HermitianTag>
		                             && has_structure_v<AntiHermitian, Hoppy::Detail::AntiHermitianTag>
		                             && !std::is_same_v<Hermitian, Symmetric>
		                             && !std::is_same_v<AntiHermitian, AntiSymmetric>
		                   : std::is_same_v<Hermitian, Symmetric>
		                             && std::is_same_v<AntiHermitian, AntiSymmetric>);
	}

	static_assert(vocabularyIsConsistent<double, Eigen::Dynamic, TrianglePacking::Lower, 0>());
	static_assert(vocabularyIsConsistent<double, 4, TrianglePacking::Upper, Eigen::DontAlign>());
	static_assert(vocabularyIsConsistent<std::complex<double>, Eigen::Dynamic,
	                                     TrianglePacking::Upper, 0>());
	static_assert(vocabularyIsConsistent<std::complex<double>, 4, TrianglePacking::Lower,
	                                     Eigen::DontAlign>());
	static_assert(std::is_same_v<Hoppy::HermitianMatrix<HoppyCustomReal>,
	                             Hoppy::SymmetricMatrix<HoppyCustomReal>>);

	static_assert(std::is_same_v<Hoppy::HermitianMatrix<double>, Hoppy::SymmetricMatrix<double>>);
	static_assert(std::is_same_v<Hoppy::AntiHermitianMatrix<double>, Hoppy::AntiSymmetricMatrix<double>>);
	static_assert(std::is_same_v<Hoppy::HermitianMatrix<float, 4, TrianglePacking::Upper, Eigen::DontAlign>,
	                             Hoppy::SymmetricMatrix<float, 4, TrianglePacking::Upper, Eigen::DontAlign>>);
	static_assert(!std::is_same_v<Hoppy::HermitianMatrix<std::complex<double>>,
	                              Hoppy::SymmetricMatrix<std::complex<double>>>);
	static_assert(!std::is_same_v<Hoppy::AntiHermitianMatrix<std::complex<double>, 4, TrianglePacking::Upper>,
	                              Hoppy::AntiSymmetricMatrix<std::complex<double>, 4, TrianglePacking::Upper>>);

	static_assert(has_structure_v<Hoppy::UpperTriangularMatrixXd, Hoppy::Detail::UpperTriangularTag>);
	static_assert(has_structure_v<Hoppy::LowerTriangularMatrixXf, Hoppy::Detail::LowerTriangularTag>);
	static_assert(has_structure_v<Hoppy::SymmetricMatrixXcd, Hoppy::Detail::SymmetricTag>);
	static_assert(has_structure_v<Hoppy::AntiSymmetricMatrixXcf, Hoppy::Detail::AntiSymmetricTag>);
	static_assert(has_structure_v<Hoppy::HermitianMatrixXcd, Hoppy::Detail::HermitianTag>);
	static_assert(has_structure_v<Hoppy::AntiHermitianMatrixXcf, Hoppy::Detail::AntiHermitianTag>);
	static_assert(std::is_same_v<typename Hoppy::UpperTriangularMatrixXi::Scalar, int>);
	static_assert(std::is_same_v<typename Hoppy::UpperTriangularMatrixXl::Scalar, long>);
	static_assert(Hoppy::LowerTriangularMatrix<double, 3>::RowsAtCompileTime == 3);
	static_assert(Hoppy::LowerTriangularMatrix<double, 3>::ColsAtCompileTime == 3);
	static_assert(Hoppy::LowerTriangularMatrixXd::RowsAtCompileTime == Eigen::Dynamic);
	static_assert(Hoppy::SymmetricMatrix<double, 4, TrianglePacking::Upper>::PackingValue
	              == TrianglePacking::Upper);

	static_assert(Hoppy::Detail::is_valid_triangular_dimension_v<Eigen::Dynamic>);
	static_assert(Hoppy::Detail::is_valid_triangular_dimension_v<0>);
	static_assert(!Hoppy::Detail::is_valid_triangular_dimension_v<-2>);
	static_assert(Hoppy::Detail::is_valid_triangular_options_v<0>);
	static_assert(Hoppy::Detail::is_valid_triangular_options_v<Eigen::AutoAlign>);
	static_assert(Hoppy::Detail::is_valid_triangular_options_v<Eigen::DontAlign>);
	static_assert(!Hoppy::Detail::is_valid_triangular_options_v<Eigen::RowMajor>);
	static_assert(!Hoppy::Detail::is_valid_triangular_options_v<4>);
	static_assert(Hoppy::Detail::is_valid_triangle_packing_v<TrianglePacking::Lower>);
	static_assert(Hoppy::Detail::is_valid_triangle_packing_v<TrianglePacking::Upper>);
	static_assert(!Hoppy::Detail::is_valid_triangle_packing_v<static_cast<TrianglePacking>(2)>);
	static_assert(has_symmetric_matrix<4, TrianglePacking::Lower, 0>::value);
	static_assert(!has_symmetric_matrix<-2, TrianglePacking::Lower, 0>::value);
	static_assert(!has_symmetric_matrix<4, TrianglePacking::Lower, Eigen::RowMajor>::value);
	static_assert(!has_symmetric_matrix<4, static_cast<TrianglePacking>(2), 0>::value);
}  // namespace

TEST_CASE("packed sizes and offsets are exact for dimensions zero through four")
{
	using namespace Hoppy::Detail;
	constexpr Eigen::Index expectedSizes[] = {0, 1, 3, 6, 10};
	for (Eigen::Index dimension = 0; dimension <= 4; ++dimension)
	{
		CAPTURE(dimension);
		REQUIRE(checkedStoredSize(dimension) == expectedSizes[dimension]);
		REQUIRE(checkedLogicalSize(dimension) == dimension * dimension);
		for (Eigen::Index row = 0; row < dimension; ++row)
			for (Eigen::Index column = 0; column < dimension; ++column)
			{
				const Eigen::Index lowerMajor = (std::max)(row, column);
				const Eigen::Index lowerMinor = (std::min)(row, column);
				const Eigen::Index upperMajor = (std::max)(row, column);
				const Eigen::Index upperMinor = (std::min)(row, column);
				REQUIRE(checkedPackedOffset(row, column, dimension, TrianglePacking::Lower)
				        == lowerMajor * (lowerMajor + 1) / 2 + lowerMinor);
				REQUIRE(checkedPackedOffset(row, column, dimension, TrianglePacking::Upper)
				        == upperMajor * (upperMajor + 1) / 2 + upperMinor);
			}
	}

	REQUIRE(checkedPackedOffset(0, 0, 4, TrianglePacking::Lower) == 0);
	REQUIRE(checkedPackedOffset(1, 0, 4, TrianglePacking::Lower) == 1);
	REQUIRE(checkedPackedOffset(1, 1, 4, TrianglePacking::Lower) == 2);
	REQUIRE(checkedPackedOffset(2, 0, 4, TrianglePacking::Lower) == 3);
	REQUIRE(checkedPackedOffset(2, 1, 4, TrianglePacking::Lower) == 4);
	REQUIRE(checkedPackedOffset(2, 2, 4, TrianglePacking::Lower) == 5);

	REQUIRE(checkedPackedOffset(0, 0, 4, TrianglePacking::Upper) == 0);
	REQUIRE(checkedPackedOffset(0, 1, 4, TrianglePacking::Upper) == 1);
	REQUIRE(checkedPackedOffset(1, 1, 4, TrianglePacking::Upper) == 2);
	REQUIRE(checkedPackedOffset(0, 2, 4, TrianglePacking::Upper) == 3);
	REQUIRE(checkedPackedOffset(1, 2, 4, TrianglePacking::Upper) == 4);
	REQUIRE(checkedPackedOffset(2, 2, 4, TrianglePacking::Upper) == 5);
}

TEST_CASE("packing canonicalization is independent of mathematical orientation")
{
	using namespace Hoppy::Detail;
	REQUIRE(canonicalPackedCoordinate(0, 3, TrianglePacking::Lower)
	        == std::make_pair(Eigen::Index{3}, Eigen::Index{0}));
	REQUIRE(canonicalPackedCoordinate(3, 0, TrianglePacking::Upper)
	        == std::make_pair(Eigen::Index{0}, Eigen::Index{3}));
	REQUIRE(checkedPackedOffset(0, 3, 4, TrianglePacking::Lower) == 6);
	REQUIRE(checkedPackedOffset(3, 0, 4, TrianglePacking::Upper) == 6);
}

TEST_CASE("valid dimension and byte-count queries")
{
	using namespace Hoppy::Detail;
	REQUIRE(checkedDimension(std::size_t{4}) == 4);
	REQUIRE(checkedSquareDimension(4, 4) == 4);
	REQUIRE(checkedStoredByteCount<double>(4) == 10 * sizeof(double));
	REQUIRE(Hoppy::SymmetricMatrix<double, 4>::requiredStoredSize(4) == 10);
	REQUIRE(Hoppy::SymmetricMatrixXd::requiredStoredSize(0) == 0);
}

#ifdef HOPPY_TEST_EIGEN_ASSERT_THROWS
TEST_CASE("invalid dimensions and independently constructed arithmetic overflows assert")
{
	using namespace Hoppy::Detail;
	using FixedSymmetric = Hoppy::SymmetricMatrix<double, 4>;
	constexpr Eigen::Index maximum = (std::numeric_limits<Eigen::Index>::max)();
	REQUIRE_THROWS_AS(checkedLogicalSize(-1), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(checkedStoredSize(-1), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(checkedLogicalSize(maximum), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(checkedStoredSize(maximum), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(checkedStoredByteCount(2, (std::numeric_limits<std::size_t>::max)()),
	                  Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(checkedSquareDimension(3, 4), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(checkedSquareDimension(-1, -1), Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(checkedPackedOffset(-1, 0, 4, TrianglePacking::Lower),
	                  Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(checkedPackedOffset(4, 0, 4, TrianglePacking::Lower),
	                  Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(checkedPackedOffset(0, 0, 4, static_cast<TrianglePacking>(2)),
	                  Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(checkedPackedOffset(maximum - 1, 0, maximum, TrianglePacking::Lower),
	                  Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(FixedSymmetric::requiredStoredSize(3),
	                  Hoppy::Test::EigenAssertionFailure);
	if constexpr ((std::numeric_limits<std::size_t>::max)()
	              > static_cast<std::size_t>((std::numeric_limits<Eigen::Index>::max)()))
	{
		REQUIRE_THROWS_AS(checkedDimension(static_cast<std::size_t>(maximum) + 1),
		                  Hoppy::Test::EigenAssertionFailure);
	}
}
#endif
