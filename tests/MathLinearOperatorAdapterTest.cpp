// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Hoppy/LowRankMatrix.hpp>
#include <SecUtility/Math/LinearOperatorAdapter.hpp>
#include <SecUtility/Math/SelfAdjointLinearOperator.hpp>

#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <concepts>
#include <type_traits>
#include <utility>


namespace
{
	template <typename Matrix>
	concept GenerallyAdaptable = requires(Matrix&& matrix) {
		SecUtility::Math::MakeLinearOperatorAdapter(std::forward<Matrix>(matrix));
	};

	template <typename Matrix>
	concept SelfAdjointlyAdaptable = requires(Matrix&& matrix) {
		SecUtility::Math::MakeSelfAdjointLinearOperatorAdapter(std::forward<Matrix>(matrix));
	};

	using General = Hoppy::LowRankMatrixX<double>;
	using RealSymmetric = Hoppy::LowRankSymmetricMatrixX<double>;
	using RealSelfAdjoint = Hoppy::LowRankSelfAdjointMatrixX<double>;
	using ComplexSelfAdjoint = Hoppy::LowRankSelfAdjointMatrixX<std::complex<double>>;

	static_assert(std::same_as<RealSymmetric, RealSelfAdjoint>);
	static_assert(GenerallyAdaptable<General>);
	static_assert(GenerallyAdaptable<Eigen::MatrixXd>);
	static_assert(SelfAdjointlyAdaptable<RealSymmetric>);
	static_assert(SelfAdjointlyAdaptable<ComplexSelfAdjoint>);
	static_assert(SelfAdjointlyAdaptable<Eigen::MatrixXcd>);
	static_assert(!GenerallyAdaptable<double>);

	using RealAdapter = decltype(
	        SecUtility::Math::MakeSelfAdjointLinearOperatorAdapter(std::declval<RealSelfAdjoint&>()));
	using ComplexAdapter = decltype(
	        SecUtility::Math::MakeSelfAdjointLinearOperatorAdapter(std::declval<Eigen::MatrixXcd&>()));
	static_assert(SecUtility::Math::SelfAdjointLinearOperator<RealAdapter>);
	static_assert(SecUtility::Math::SelfAdjointLinearOperator<ComplexAdapter>);
	static_assert(std::same_as<typename ComplexAdapter::RealScalar, double>);
}


TEST_CASE("Linear operator adapters apply dense and low-rank matrices", "[Math][LinearOperatorAdapter]")
{
	General lowRank(3, 2);
	Eigen::Vector3d left;
	left << 1.0, -2.0, 0.5;
	Eigen::Vector2d right;
	right << 3.0, -1.0;
	lowRank.addTerm(2.0, left, right);
	const Eigen::Matrix<double, 3, 2> dense = lowRank.toDense();
	const auto lowRankAdapter = SecUtility::Math::MakeLinearOperatorAdapter(lowRank);
	const auto denseAdapter = SecUtility::Math::MakeLinearOperatorAdapter(dense);
	const Eigen::Vector2d vector(4.0, -2.0);
	Eigen::Matrix<double, 2, 2> block;
	block << 4.0, 1.0, -2.0, 3.0;

	CHECK(lowRankAdapter.rows() == 3);
	CHECK(lowRankAdapter.cols() == 2);
	CHECK(lowRankAdapter.ApplyOn(vector).isApprox(dense * vector));
	CHECK(lowRankAdapter.ApplyOn(block).isApprox(dense * block));
	CHECK(denseAdapter.ApplyOn(vector).isApprox(dense * vector));
	CHECK(denseAdapter.ApplyOn(block).isApprox(dense * block));
	CHECK((lowRank * vector).eval().isApprox(lowRankAdapter.ApplyOn(vector)));
}


TEST_CASE("Self-adjoint adapters expose real diagonals for dense and low-rank sources",
	          "[Math][LinearOperatorAdapter]")
{
	using Scalar = std::complex<double>;
	ComplexSelfAdjoint lowRank(3);
	Eigen::VectorX<Scalar> vector(3);
	vector << Scalar{1.0, 2.0}, Scalar{-2.0, 0.5}, Scalar{0.25, -1.0};
	lowRank.addTerm(1.5, vector);
	const Eigen::MatrixX<Scalar> dense = lowRank.toDense();
	const auto lowRankAdapter = SecUtility::Math::MakeSelfAdjointLinearOperatorAdapter(lowRank);
	const auto denseAdapter = SecUtility::Math::MakeSelfAdjointLinearOperatorAdapter(dense);
	const Eigen::MatrixX<Scalar> block = Eigen::MatrixX<Scalar>::Random(3, 2);

	static_assert(std::same_as<std::remove_cvref_t<decltype(lowRankAdapter.Diagonal())>, Eigen::VectorXd>);
	CHECK(lowRankAdapter.Diagonal().isApprox(dense.diagonal().real()));
	CHECK(denseAdapter.Diagonal().isApprox(dense.diagonal().real()));
	CHECK(lowRankAdapter.ApplyOn(block).isApprox(dense * block));
	CHECK(denseAdapter.ApplyOn(block).isApprox(dense * block));
	CHECK(SecUtility::Math::ApplySelfAdjointLinearOperator(lowRankAdapter, block).isApprox(dense * block));
}


TEST_CASE("Dense plus low-rank updates use the general adapter", "[Math][LinearOperatorAdapter]")
{
	Eigen::Matrix3d initial;
	initial << 4.0, 1.0, 0.0, 1.0, 3.0, -1.0, 0.0, -1.0, 2.0;
	RealSelfAdjoint update(3);
	Eigen::Vector3d direction;
	direction << 1.0, -0.5, 2.0;
	update.addTerm(0.25, direction);
	auto expression = initial + update;
	const auto adapter = SecUtility::Math::MakeSelfAdjointLinearOperatorAdapter(expression);
	const Eigen::MatrixXd block = Eigen::MatrixXd::Random(3, 4);

	static_assert(SecUtility::Math::SelfAdjointLinearOperator<decltype(adapter)>);
	CHECK(adapter.Diagonal().isApprox(expression.diagonal()));
	CHECK(adapter.ApplyOn(block).isApprox(expression.toDense() * block));
}


TEST_CASE("Adapters own temporary expressions and reference lvalues", "[Math][LinearOperatorAdapter]")
{
	Eigen::Matrix2d dense;
	dense << 2.0, 1.0, 1.0, 3.0;
	const auto referencing = SecUtility::Math::MakeSelfAdjointLinearOperatorAdapter(dense);
	const auto owningPlain = SecUtility::Math::MakeLinearOperatorAdapter(Eigen::Matrix2d{dense});
	const auto owningDense = SecUtility::Math::MakeLinearOperatorAdapter(dense + Eigen::Matrix2d::Identity());
	RealSelfAdjoint update(2);
	Eigen::Vector2d direction(1.0, -2.0);
	update.addTerm(0.5, direction);
	const auto owningComposite =
	        SecUtility::Math::MakeSelfAdjointLinearOperatorAdapter(Eigen::Matrix2d::Identity() + update);
	const Eigen::Vector2d vector(2.0, 3.0);

	static_assert(std::is_reference_v<typename decltype(referencing)::Nested>);
	static_assert(!std::is_reference_v<typename decltype(owningPlain)::Nested>);
	static_assert(!std::is_reference_v<typename decltype(owningDense)::Nested>);
	static_assert(!std::is_reference_v<typename decltype(owningComposite)::Nested>);
	CHECK(referencing.ApplyOn(vector).isApprox(dense * vector));
	CHECK(owningPlain.ApplyOn(vector).isApprox(dense * vector));
	CHECK(owningDense.ApplyOn(vector).isApprox((dense + Eigen::Matrix2d::Identity()) * vector));
	CHECK(owningComposite.ApplyOn(vector).isApprox((Eigen::Matrix2d::Identity() + update.toDense()) * vector));
}
