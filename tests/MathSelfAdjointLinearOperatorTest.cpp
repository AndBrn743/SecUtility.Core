// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Math/SelfAdjointLinearOperator.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <type_traits>


namespace
{
	template <typename T>
	struct BlockOperator
	{
		using Scalar = T;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		Eigen::MatrixX<Scalar> Matrix;
		mutable Eigen::Index ApplicationCount = 0;

		Eigen::Index rows() const { return Matrix.rows(); }
		Eigen::Index cols() const { return Matrix.cols(); }
		Eigen::VectorX<RealScalar> Diagonal() const { return Matrix.diagonal().real(); }

		template <typename Derived>
		Eigen::MatrixX<Scalar> ApplyOn(const Eigen::MatrixBase<Derived>& vectors) const
		{
			ApplicationCount++;
			return Matrix * vectors;
		}
	};


	template <typename T>
	struct VectorOnlyOperator
	{
		using Scalar = T;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		Eigen::MatrixX<Scalar> Matrix;
		mutable Eigen::Index ApplicationCount = 0;

		Eigen::Index rows() const { return Matrix.rows(); }
		Eigen::Index cols() const { return Matrix.cols(); }
		Eigen::VectorX<RealScalar> Diagonal() const { return Matrix.diagonal().real(); }

		template <typename Derived>
			requires(Derived::ColsAtCompileTime == 1)
		Eigen::VectorX<Scalar> ApplyOn(const Eigen::MatrixBase<Derived>& vector) const
		{
			ApplicationCount++;
			return Matrix * vector;
		}
	};


	struct MissingScalar
	{
		Eigen::Index rows() const;
		Eigen::Index cols() const;
		Eigen::VectorXd Diagonal() const;
		Eigen::VectorXd ApplyOn(const Eigen::VectorXd&) const;
	};


	struct MissingDiagonal
	{
		using Scalar = double;
		Eigen::Index rows() const;
		Eigen::Index cols() const;
		Eigen::VectorXd ApplyOn(const Eigen::VectorXd&) const;
	};


	struct ComplexDiagonal
	{
		using Scalar = std::complex<double>;
		Eigen::Index rows() const;
		Eigen::Index cols() const;
		Eigen::VectorXcd Diagonal() const;
		Eigen::VectorXcd ApplyOn(const Eigen::VectorXcd&) const;
	};


	template <typename T>
	struct NoApplication
	{
		using Scalar = T;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		Eigen::Index rows() const;
		Eigen::Index cols() const;
		Eigen::VectorX<RealScalar> Diagonal() const;
	};
}


using namespace SecUtility::Math;

static_assert(std::same_as<LinearOperatorScalar<const BlockOperator<double>&>, double>);
static_assert(std::same_as<LinearOperatorRealScalar<BlockOperator<std::complex<float>>>, float>);
static_assert(SelfAdjointLinearOperator<BlockOperator<double>>);
static_assert(SelfAdjointLinearOperator<BlockOperator<std::complex<double>>>);
static_assert(BlockSelfAdjointLinearOperator<BlockOperator<double>>);
static_assert(ScalarSelfAdjointLinearOperator<BlockOperator<double>>);
static_assert(SelfAdjointLinearOperator<VectorOnlyOperator<double>>);
static_assert(SelfAdjointLinearOperator<VectorOnlyOperator<std::complex<double>>>);
static_assert(ScalarSelfAdjointLinearOperator<VectorOnlyOperator<double>>);
static_assert(!BlockSelfAdjointLinearOperator<VectorOnlyOperator<double>>);
static_assert(!SelfAdjointLinearOperator<MissingScalar>);
static_assert(!SelfAdjointLinearOperator<MissingDiagonal>);
static_assert(!SelfAdjointLinearOperator<ComplexDiagonal>);
static_assert(!SelfAdjointLinearOperator<NoApplication<double>>);
static_assert(!SelfAdjointLinearOperator<BlockOperator<int>>);
static_assert(!SelfAdjointLinearOperator<BlockOperator<std::complex<int>>>);


TEMPLATE_TEST_CASE("Self-adjoint operator dispatch uses one block application", "[Math][SelfAdjointLinearOperator]",
	               double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> matrix(3, 3);
	matrix << TestType{2}, TestType{1}, TestType{0}, TestType{1}, TestType{3}, TestType{1}, TestType{0}, TestType{1},
	        TestType{4};
	BlockOperator<TestType> linearOperator{matrix};
	const Eigen::MatrixX<TestType> vectors = Eigen::MatrixX<TestType>::Random(3, 4);

	const auto images = ApplySelfAdjointLinearOperator(linearOperator, vectors);

	CHECK(images.isApprox(matrix * vectors));
	CHECK(linearOperator.ApplicationCount == 1);
}


TEMPLATE_TEST_CASE("Self-adjoint operator dispatch falls back to individual columns",
	               "[Math][SelfAdjointLinearOperator]", double, (std::complex<double>))
{
	Eigen::MatrixX<TestType> matrix(3, 3);
	matrix << TestType{2}, TestType{1}, TestType{0}, TestType{1}, TestType{3}, TestType{1}, TestType{0}, TestType{1},
	        TestType{4};
	VectorOnlyOperator<TestType> linearOperator{matrix};
	const Eigen::MatrixX<TestType> vectors = Eigen::MatrixX<TestType>::Random(3, 4);

	const auto images = ApplySelfAdjointLinearOperator(linearOperator, vectors);

	CHECK(images.isApprox(matrix * vectors));
	CHECK(linearOperator.ApplicationCount == vectors.cols());
}


TEMPLATE_TEST_CASE("Empty self-adjoint operator applications preserve dimensions and do not invoke the operator",
	               "[Math][SelfAdjointLinearOperator]", double, (std::complex<double>))
{
	BlockOperator<TestType> blockOperator{Eigen::MatrixX<TestType>::Identity(3, 3)};
	VectorOnlyOperator<TestType> vectorOperator{Eigen::MatrixX<TestType>::Identity(3, 3)};
	const Eigen::MatrixX<TestType> emptyVectors(3, 0);

	const auto blockImages = ApplySelfAdjointLinearOperator(blockOperator, emptyVectors);
	const auto vectorImages = ApplySelfAdjointLinearOperator(vectorOperator, emptyVectors);

	CHECK(blockImages.rows() == 3);
	CHECK(blockImages.cols() == 0);
	CHECK(vectorImages.rows() == 3);
	CHECK(vectorImages.cols() == 0);
	CHECK(blockOperator.ApplicationCount == 0);
	CHECK(vectorOperator.ApplicationCount == 0);
}
