// SPDX-License-Identifier: MIT

// Triangular-compressed specification: sections 10-11, 14, and 17.

#include "HoppyTestSupport.hpp"

#include <SecUtility/Hoppy/TriangularCompressedMatrix.hpp>

#include <Eigen/Core>

#include <array>
#include <complex>
#include <type_traits>
#include <utility>

namespace
{
	template <typename T, typename = void> struct has_rvalue_diagonal : std::false_type {};
	template <typename T>
	struct has_rvalue_diagonal<T, std::void_t<decltype(std::declval<T&&>().diagonal())>> : std::true_type {};
	template <typename T, typename = void> struct has_all_rvalue_views : std::false_type {};
	template <typename T>
	struct has_all_rvalue_views<T, std::void_t<
	        decltype(std::declval<T&&>().diagonal()),
	        decltype(std::declval<T&&>().template triangularView<Eigen::StrictlyUpper>()),
	        decltype(std::declval<T&&>().block(0, 0, 2, 2)),
	        decltype(std::declval<T&&>().template block<2, 2>(0, 0)),
	        decltype(std::declval<T&&>().topLeftCorner(2, 2)),
	        decltype(std::declval<T&&>().template topLeftCorner<2, 2>()),
	        decltype(std::declval<T&&>().row(0)),
	        decltype(std::declval<T&&>().col(0))>> : std::true_type {};

	using Upper = Hoppy::UpperTriangularMatrix<std::complex<double>, 3, Hoppy::TrianglePacking::Lower>;
	using Transpose = decltype(std::declval<const Upper&>().transpose());
	using Conjugate = decltype(std::declval<const Upper&>().conjugate());
	using Adjoint = decltype(std::declval<const Upper&>().adjoint());
	static_assert(std::is_same_v<typename Transpose::StructureTag, Hoppy::Detail::LowerTriangularTag>);
	static_assert(std::is_same_v<typename Conjugate::StructureTag, Hoppy::Detail::UpperTriangularTag>);
	static_assert(std::is_same_v<typename Adjoint::StructureTag, Hoppy::Detail::LowerTriangularTag>);
	static_assert(Transpose::PackingValue == Hoppy::TrianglePacking::Upper);
	static_assert(Conjugate::PackingValue == Hoppy::TrianglePacking::Lower);
	static_assert(Adjoint::PackingValue == Hoppy::TrianglePacking::Upper);
	static_assert(has_rvalue_diagonal<Upper>::value);
	static_assert(has_all_rvalue_views<Upper>::value);
	using MutableMap = Eigen::Map<Upper>;
	using ConstMap = Eigen::Map<const Upper>;
	static_assert(has_all_rvalue_views<MutableMap>::value);
	static_assert(has_all_rvalue_views<ConstMap>::value);
	static_assert(has_all_rvalue_views<Transpose>::value);
	static_assert((Eigen::internal::traits<Transpose>::Flags & Eigen::DirectAccessBit) == 0);
	static_assert(Eigen::internal::evaluator<Transpose>::Flags == 0);
	using LowerTranspose = decltype(std::declval<const Hoppy::LowerTriangularMatrixXd&>().transpose());
	using SymmetricTranspose = decltype(std::declval<const Hoppy::SymmetricMatrixXd&>().transpose());
	using AntiTranspose = decltype(std::declval<const Hoppy::AntiSymmetricMatrixXd&>().transpose());
	using HermitianTranspose = decltype(std::declval<const Hoppy::HermitianMatrixXcd&>().transpose());
	using AntiHermitianAdjoint = decltype(std::declval<const Hoppy::AntiHermitianMatrixXcd&>().adjoint());
	static_assert(std::is_same_v<typename LowerTranspose::StructureTag, Hoppy::Detail::UpperTriangularTag>);
	static_assert(std::is_same_v<typename SymmetricTranspose::StructureTag, Hoppy::Detail::SymmetricTag>);
	static_assert(std::is_same_v<typename AntiTranspose::StructureTag, Hoppy::Detail::AntiSymmetricTag>);
	static_assert(std::is_same_v<typename HermitianTranspose::StructureTag, Hoppy::Detail::HermitianTag>);
	static_assert(std::is_same_v<typename AntiHermitianAdjoint::StructureTag, Hoppy::Detail::AntiHermitianTag>);
}

TEST_CASE("transpose conjugate and adjoint are lazy structured expressions")
{
	using Complex = std::complex<double>;
	Upper matrix;
	matrix.setZero();
	matrix(0, 1) = Complex(2, 3);
	matrix(1, 2) = Complex(4, -5);
	const auto dense = matrix.toDense();
	Hoppy::Test::requireApprox(matrix.transpose().toDense(), dense.transpose());
	Hoppy::Test::requireApprox(matrix.conjugate().toDense(), dense.conjugate());
	Hoppy::Test::requireApprox(matrix.adjoint().toDense(), dense.adjoint());
	Hoppy::Test::requireApprox(matrix.transpose().transpose().toDense(), dense);
	Hoppy::Test::requireApprox(matrix.transpose().conjugate().toDense(), dense.transpose().conjugate());

	const auto evaluated = matrix.adjoint().eval();
	static_assert(std::is_same_v<typename std::decay_t<decltype(evaluated)>::StructureTag,
	                             Hoppy::Detail::LowerTriangularTag>);
	REQUIRE(evaluated.coeff(1, 0) == Complex(2, -3));
	const auto transposed = matrix.transpose();
	Eigen::internal::evaluator<decltype(transposed)> evaluator(transposed);
	REQUIRE(evaluator.coeff(1, 0) == Complex(2, 3));
	Eigen::MatrixXcd implicit = matrix.adjoint();
	REQUIRE(implicit(1, 0) == Complex(2, -3));
}

TEST_CASE("diagonal and logical subviews route writes through checked proxies")
{
	Hoppy::SymmetricMatrix<double, 4> matrix;
	matrix.setZero();
	auto diagonal = matrix.diagonal();
	diagonal[2] = 7;
	REQUIRE(matrix.coeff(2, 2) == 7);

	auto block = matrix.block(1, 0, 2, 3);
	block(0, 2) = 5;
	REQUIRE(matrix.coeff(1, 2) == 5);
	REQUIRE(matrix.coeff(2, 1) == 5);
	auto corner = matrix.topLeftCorner(2, 2);
	corner(0, 1) = 3;
	REQUIRE(matrix.coeff(1, 0) == 3);
	matrix.row(3)(0, 1) = 9;
	REQUIRE(matrix.coeff(3, 1) == 9);
	matrix.col(0)(2, 0) = 8;
	REQUIRE(matrix.coeff(0, 2) == 8);
}

TEST_CASE("rvalue views support immediate Eigen-style consumption")
{
	using Matrix = Hoppy::SymmetricMatrix<double, 3>;
	Hoppy::Test::requireApprox(Matrix::Identity().diagonal().toDense(), Eigen::Vector3d::Ones());
	Hoppy::Test::requireApprox(Matrix::Identity().template triangularView<Eigen::Upper>().toDense(),
	                           Eigen::Matrix3d::Identity());
	Hoppy::Test::requireApprox(Matrix::Identity().block(0, 0, 2, 2).toDense(),
	                           Eigen::Matrix2d::Identity());
	Hoppy::Test::requireApprox(Matrix::Identity().template topLeftCorner<2, 2>().toDense(),
	                           Eigen::Matrix2d::Identity());
	Hoppy::Test::requireApprox(Matrix::Identity().row(1).toDense(),
	                           Eigen::RowVector3d(0, 1, 0));
	Hoppy::Test::requireApprox(Matrix::Identity().col(1).toDense(), Eigen::Vector3d(0, 1, 0));
	Hoppy::Test::requireApprox(Matrix::Identity().transpose().diagonal().toDense(),
	                           Eigen::Vector3d::Ones());
}

TEST_CASE("all rvalue logical views agree with nontrivial dense oracles")
{
	using Matrix = Hoppy::SymmetricMatrix<double, 3>;
	Eigen::Matrix3d dense;
	dense << 2, 3, 5,
	         3, 7, 11,
	         5, 11, 13;
	const auto makeMatrix = [&] { return Matrix::FromUncheckedDense(dense); };

	Hoppy::Test::requireApprox(makeMatrix().diagonal().toDense(), dense.diagonal());
	Hoppy::Test::requireApprox(makeMatrix().template triangularView<Eigen::Upper>().toDense(),
	                           dense.template triangularView<Eigen::Upper>().toDenseMatrix());
	Hoppy::Test::requireApprox(makeMatrix().template triangularView<Eigen::StrictlyLower>().toDense(),
	                           dense.template triangularView<Eigen::StrictlyLower>().toDenseMatrix());
	Hoppy::Test::requireApprox(makeMatrix().block(1, 0, 2, 2).toDense(), dense.block(1, 0, 2, 2));
	Hoppy::Test::requireApprox(makeMatrix().template block<2, 2>(0, 1).toDense(),
	                           dense.template block<2, 2>(0, 1));
	Hoppy::Test::requireApprox(makeMatrix().topLeftCorner(2, 2).toDense(),
	                           dense.topLeftCorner(2, 2));
	Hoppy::Test::requireApprox(makeMatrix().template topLeftCorner<2, 2>().toDense(),
	                           dense.template topLeftCorner<2, 2>());
	Hoppy::Test::requireApprox(makeMatrix().row(2).toDense(), dense.row(2));
	Hoppy::Test::requireApprox(makeMatrix().col(1).toDense(), dense.col(1));
}

TEST_CASE("rvalue expression views remain valid through immediate nested consumption")
{
	Hoppy::SymmetricMatrix<double, 3> matrix;
	matrix.setZero();
	matrix(0, 0) = 2; matrix(1, 1) = 3; matrix(2, 2) = 4;
	matrix(0, 1) = 5; matrix(0, 2) = 7; matrix(1, 2) = 11;
	const Eigen::Matrix3d dense = matrix.toDense();

	Hoppy::Test::requireApprox((matrix + matrix).diagonal().toDense(),
	                           (dense + dense).diagonal());
	Hoppy::Test::requireApprox((matrix + matrix).block(0, 1, 2, 2).toDense(),
	                           (dense + dense).block(0, 1, 2, 2));
	Hoppy::Test::requireApprox(matrix.transpose().template triangularView<Eigen::Lower>().toDense(),
	                           dense.transpose().template triangularView<Eigen::Lower>().toDenseMatrix());
	Hoppy::Test::requireApprox(matrix.conjugate().template topLeftCorner<2, 2>().toDense(),
	                           dense.template topLeftCorner<2, 2>());
	Hoppy::Test::requireApprox(matrix.adjoint().row(1).toDense(), dense.adjoint().row(1));
	Hoppy::Test::requireApprox(matrix.inverse().col(0).toDense(), dense.inverse().col(0));
}

TEST_CASE("rvalue mutable map views preserve write-through behavior")
{
	std::array<double, 6> storage{};
	Eigen::Map<Hoppy::SymmetricMatrix<double, 3>> map(storage.data());
	map.setZero();
	std::move(map).diagonal()[0] = 2;
	std::move(map).block(0, 0, 2, 2)(0, 1) = 3;
	std::move(map).template block<2, 2>(1, 1)(1, 1) = 5;
	std::move(map).topLeftCorner(2, 2)(1, 1) = 7;
	std::move(map).template topLeftCorner<2, 2>()(0, 0) = 11;
	std::move(map).row(2)(0, 0) = 13;
	std::move(map).col(2)(1, 0) = 17;
	std::move(map).template triangularView<Eigen::Upper>()(0, 2) = 19;

	REQUIRE(map.coeff(0, 0) == 11);
	REQUIRE(map.coeff(0, 1) == 3);
	REQUIRE(map.coeff(1, 1) == 7);
	REQUIRE(map.coeff(1, 2) == 17);
	REQUIRE(map.coeff(2, 0) == 19);
	REQUIRE(map.coeff(2, 2) == 5);
}

TEST_CASE("rvalue const map views remain read-only and immediately consumable")
{
	const std::array<double, 6> storage{2, 3, 5, 7, 11, 13};
	Eigen::Map<const Hoppy::SymmetricMatrix<double, 3>> map(storage.data());
	const auto dense = map.toDense();
	Hoppy::Test::requireApprox(std::move(map).diagonal().toDense(), dense.diagonal());
	Hoppy::Test::requireApprox(std::move(map).block(0, 1, 3, 2).toDense(), dense.block(0, 1, 3, 2));
}

TEST_CASE("triangular logical views expose implicit zeros and checked writes")
{
	Hoppy::SymmetricMatrix<double, 3> matrix;
	matrix.setZero();
	auto upper = matrix.triangularView<Eigen::Upper>();
	upper(0, 2) = 6;
	REQUIRE(upper.coeff(0, 2) == 6);
	REQUIRE(upper.coeff(2, 0) == 0);
	REQUIRE(matrix.coeff(2, 0) == 6);

	const auto& constant = matrix;
	auto strictLower = constant.triangularView<Eigen::StrictlyLower>();
	REQUIRE(strictLower.coeff(2, 0) == 6);
	REQUIRE(strictLower.coeff(0, 2) == 0);
	REQUIRE(strictLower.coeff(1, 1) == 0);
}

TEST_CASE("maps expose the same transform and logical view surface")
{
	using Complex = std::complex<double>;
	std::array<Complex, 6> storage{};
	Eigen::Map<Hoppy::HermitianMatrix<Complex, 3>> map(storage.data());
	map.setZero();
	map(0, 2) = Complex(2, 5);
	map.diagonal()[1] = Complex(7, 0);
	REQUIRE(map.block(0, 1, 2, 2).coeff(0, 1) == Complex(2, 5));
	REQUIRE(map.adjoint().coeff(2, 0) == Complex(2, -5));

	Eigen::Map<const Hoppy::HermitianMatrix<Complex, 3>> constant(storage.data());
	REQUIRE(constant.col(2).coeff(0, 0) == Complex(2, 5));
	Eigen::internal::evaluator<decltype(constant)> evaluator(constant);
	REQUIRE(evaluator.coeff(2, 0) == Complex(2, -5));
}

#ifdef HOPPY_TEST_EIGEN_ASSERT_THROWS
TEST_CASE("view bounds and structural constraints assert")
{
	Hoppy::AntiSymmetricMatrix<double, 3> anti;
	anti.setZero();
	REQUIRE_THROWS_AS(anti.diagonal()[1] = 2, Hoppy::Test::EigenAssertionFailure);
	REQUIRE_THROWS_AS(anti.block(2, 2, 2, 1), Hoppy::Test::EigenAssertionFailure);
	auto upper = anti.triangularView<Eigen::Upper>();
	REQUIRE_THROWS_AS(upper(2, 0) = 4, Hoppy::Test::EigenAssertionFailure);
}
#endif
