// SPDX-License-Identifier: MIT

// Triangular-compressed specification: D13-D14; sections 8 and 14.

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/TriangularCompressedMatrix.hpp>

#include <Eigen/Core>

#include <array>
#include <complex>
#include <type_traits>

namespace
{
	using Dense = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
	using UpperView = decltype(std::declval<Dense&>().template triangularView<Eigen::Upper>());
	using LowerView = decltype(std::declval<Dense&>().template triangularView<Eigen::Lower>());
	using StrictUpperView = decltype(std::declval<Dense&>().template triangularView<Eigen::StrictlyUpper>());
	using StrictLowerView = decltype(std::declval<Dense&>().template triangularView<Eigen::StrictlyLower>());
	using UnitUpperView = decltype(std::declval<Dense&>().template triangularView<Eigen::UnitUpper>());
	using UnitLowerView = decltype(std::declval<Dense&>().template triangularView<Eigen::UnitLower>());
	using SelfAdjoint = decltype(std::declval<Dense&>().template selfadjointView<Eigen::Upper>());

	static_assert(std::is_assignable_v<Hoppy::UpperTriangularMatrixXd&, UpperView>);
	static_assert(std::is_assignable_v<Hoppy::UpperTriangularMatrixXd&, StrictUpperView>);
	static_assert(std::is_assignable_v<Hoppy::UpperTriangularMatrixXd&, UnitUpperView>);
	static_assert(!std::is_assignable_v<Hoppy::UpperTriangularMatrixXd&, LowerView>);
	static_assert(!std::is_assignable_v<Hoppy::UpperTriangularMatrixXd&, StrictLowerView>);
	static_assert(!std::is_assignable_v<Hoppy::UpperTriangularMatrixXd&, UnitLowerView>);
	static_assert(std::is_assignable_v<Hoppy::LowerTriangularMatrixXd&, LowerView>);
	static_assert(std::is_assignable_v<Hoppy::LowerTriangularMatrixXd&, StrictLowerView>);
	static_assert(std::is_assignable_v<Hoppy::LowerTriangularMatrixXd&, UnitLowerView>);
	static_assert(!std::is_assignable_v<Hoppy::LowerTriangularMatrixXd&, UpperView>);
	static_assert(!std::is_assignable_v<Hoppy::LowerTriangularMatrixXd&, StrictUpperView>);
	static_assert(!std::is_assignable_v<Hoppy::LowerTriangularMatrixXd&, UnitUpperView>);
	static_assert(std::is_assignable_v<Hoppy::SymmetricMatrixXd&, UpperView>);
	static_assert(std::is_assignable_v<Hoppy::SymmetricMatrixXd&, LowerView>);
	static_assert(std::is_assignable_v<Hoppy::SymmetricMatrixXd&, StrictUpperView>);
	static_assert(std::is_assignable_v<Hoppy::SymmetricMatrixXd&, StrictLowerView>);
	static_assert(std::is_assignable_v<Hoppy::SymmetricMatrixXd&, UnitUpperView>);
	static_assert(std::is_assignable_v<Hoppy::SymmetricMatrixXd&, UnitLowerView>);
	static_assert(std::is_assignable_v<Hoppy::HermitianMatrixXcd&, UpperView>);
	static_assert(std::is_assignable_v<Hoppy::HermitianMatrixXcd&, LowerView>);
	static_assert(std::is_assignable_v<Hoppy::HermitianMatrixXcd&, StrictUpperView>);
	static_assert(std::is_assignable_v<Hoppy::HermitianMatrixXcd&, StrictLowerView>);
	static_assert(std::is_assignable_v<Hoppy::HermitianMatrixXcd&, UnitUpperView>);
	static_assert(std::is_assignable_v<Hoppy::HermitianMatrixXcd&, UnitLowerView>);
	static_assert(std::is_assignable_v<Hoppy::AntiSymmetricMatrixXd&, UpperView>);
	static_assert(std::is_assignable_v<Hoppy::AntiSymmetricMatrixXd&, LowerView>);
	static_assert(std::is_assignable_v<Hoppy::AntiSymmetricMatrixXd&, StrictUpperView>);
	static_assert(std::is_assignable_v<Hoppy::AntiSymmetricMatrixXd&, StrictLowerView>);
	static_assert(!std::is_assignable_v<Hoppy::AntiSymmetricMatrixXd&, UnitUpperView>);
	static_assert(!std::is_assignable_v<Hoppy::AntiSymmetricMatrixXd&, UnitLowerView>);
	static_assert(std::is_assignable_v<Hoppy::AntiHermitianMatrixXcd&, UpperView>);
	static_assert(std::is_assignable_v<Hoppy::AntiHermitianMatrixXcd&, LowerView>);
	static_assert(std::is_assignable_v<Hoppy::AntiHermitianMatrixXcd&, StrictUpperView>);
	static_assert(std::is_assignable_v<Hoppy::AntiHermitianMatrixXcd&, StrictLowerView>);
	static_assert(!std::is_assignable_v<Hoppy::AntiHermitianMatrixXcd&, UnitUpperView>);
	static_assert(!std::is_assignable_v<Hoppy::AntiHermitianMatrixXcd&, UnitLowerView>);
	static_assert(std::is_assignable_v<Hoppy::HermitianMatrixXd&, SelfAdjoint>);
	static_assert(std::is_assignable_v<Hoppy::SymmetricMatrixXd&, SelfAdjoint>);
	static_assert(!std::is_assignable_v<Hoppy::UpperTriangularMatrixXd&, SelfAdjoint>);
	static_assert(!std::is_assignable_v<Hoppy::SymmetricMatrixXcd&,
	                                   decltype(std::declval<Eigen::MatrixXcd&>()
	                                                    .template selfadjointView<Eigen::Upper>())>);
	static_assert(!std::is_assignable_v<Hoppy::SymmetricMatrixXd&, Dense>);
	static_assert(!std::is_constructible_v<Hoppy::SymmetricMatrixXd, Dense>);
	static_assert(std::is_constructible_v<Dense, Hoppy::SymmetricMatrixXd>);
	static_assert(std::is_assignable_v<Dense&, Hoppy::SymmetricMatrixXd>);
}

TEST_CASE("triangular views provide their selected implicit diagonal")
{
	Dense source(3, 3);
	source << 3, 2, 4, 8, 5, 6, 7, 9, 1;
	Hoppy::SymmetricMatrixXd strict;
	strict = source.triangularView<Eigen::StrictlyUpper>();
	REQUIRE(strict.dimension() == 3);
	REQUIRE(strict.coeff(0, 2) == 4);
	REQUIRE(strict.coeff(2, 0) == 4);
	REQUIRE(strict.coeff(1, 1) == 0);

	Hoppy::LowerTriangularMatrixXd unit;
	unit = source.triangularView<Eigen::UnitLower>();
	REQUIRE(unit.coeff(2, 0) == 7);
	REQUIRE(unit.coeff(1, 1) == 1);
	REQUIRE(unit.coeff(0, 2) == 0);
}

TEST_CASE("self-adjoint views and opposite packing structured copies are supported")
{
	Dense source(2, 2);
	source << 1, 3, 9, 2;
	Hoppy::SymmetricMatrix<double, Eigen::Dynamic, Hoppy::TrianglePacking::Upper> upperPacked;
	upperPacked = source.selfadjointView<Eigen::Upper>();
	Hoppy::SymmetricMatrix<double, Eigen::Dynamic, Hoppy::TrianglePacking::Lower> lowerPacked;
	lowerPacked = upperPacked;
	REQUIRE(lowerPacked.coeff(0, 1) == 3);
	REQUIRE(lowerPacked.coeff(1, 0) == 3);
}

TEST_CASE("maps accept alias-safe view and structured assignment")
{
	std::array<double, 6> storage{};
	Eigen::Map<Hoppy::SymmetricMatrix<double, 3>> map(storage.data());
	Dense source = Dense::Constant(3, 3, 4.0);
	map = source.triangularView<Eigen::Lower>();
	REQUIRE(map.coeff(0, 2) == 4.0);
	Eigen::Map<Hoppy::SymmetricMatrix<double, 3>>& sameMap = map;
	map = sameMap;
	REQUIRE(map.coeff(0, 2) == 4.0);
}

TEST_CASE("maps accept self-adjoint views and reject dimension mismatches before mutation")
{
	std::array<double, 6> storage{};
	Eigen::Map<Hoppy::SymmetricMatrix<double, 3>> map(storage.data());
	Dense source(3, 3);
	source << 1, 2, 3, 20, 4, 5, 30, 50, 6;
	map = source.selfadjointView<Eigen::Upper>();
	REQUIRE(map.coeff(0, 2) == 3.0);
	REQUIRE(map.coeff(2, 0) == 3.0);
	REQUIRE(map.coeff(1, 1) == 4.0);

	const auto before = map.toDense();
	Dense wrongSize = Dense::Constant(2, 2, 9.0);
#ifndef EIGEN_NO_DEBUG
	REQUIRE_THROWS_AS(map = wrongSize.selfadjointView<Eigen::Lower>(),
	                  Hoppy::Test::EigenAssertionFailure);
#else
	map = wrongSize.selfadjointView<Eigen::Lower>();
#endif
	Hoppy::Test::requireApprox(map.toDense(), before);
}

TEST_CASE("dense extraction supports fixed dynamic and row-major destinations")
{
	auto matrix = Hoppy::AntiSymmetricMatrix<double, 3>::Zero();
	matrix(0, 2) = 5;
	const Eigen::Matrix3d fixed = matrix.toDense();
	REQUIRE(fixed(2, 0) == -5);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rowMajor;
	matrix.evalTo(rowMajor);
	REQUIRE(rowMajor.rows() == 3);
	REQUIRE(rowMajor(0, 2) == 5);
	REQUIRE(rowMajor(2, 0) == -5);
	Eigen::MatrixXd implicit = matrix;
	Eigen::MatrixXd assigned;
	assigned = matrix;
	REQUIRE(implicit(2, 0) == -5);
	REQUIRE(assigned(0, 2) == 5);
}
