// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include <SecUtility/Math/IterativeVectorInteractionSelfAdjointEigenSolver.hpp>
#include <SecUtility/Math/MatrixFreeLinearOperator.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <limits>
#include <type_traits>


namespace
{
	template <typename T>
	struct DenseSelfAdjointLinearOperator
	{
		using Scalar = T;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

		Eigen::MatrixX<Scalar> Matrix;

		[[nodiscard]] Eigen::Index rows() const noexcept
		{
			return Matrix.rows();
		}

		[[nodiscard]] Eigen::Index cols() const noexcept
		{
			return Matrix.cols();
		}

		[[nodiscard]] Eigen::VectorX<RealScalar> Diagonal() const
		{
			return Matrix.diagonal().real();
		}

		template <typename Derived>
		[[nodiscard]] auto ApplyOn(const Eigen::MatrixBase<Derived>& vectors) const
		{
			return (Matrix * vectors).eval();
		}
	};


	template <typename Scalar>
	DenseSelfAdjointLinearOperator<Scalar> IdentityOperator(const Eigen::Index dimension)
	{
		return {Eigen::MatrixX<Scalar>::Identity(dimension, dimension)};
	}


	struct OperatorWithoutDiagonal
	{
		using Scalar = double;

		[[nodiscard]] Eigen::Index rows() const noexcept { return 3; }
		[[nodiscard]] Eigen::Index cols() const noexcept { return 3; }
		[[nodiscard]] Eigen::VectorXd ApplyOn(const Eigen::VectorXd& vector) const { return vector; }
	};


	struct ComplexDiagonalOperator
	{
		using Scalar = std::complex<double>;

		[[nodiscard]] Eigen::Index rows() const noexcept { return 3; }
		[[nodiscard]] Eigen::Index cols() const noexcept { return 3; }
		[[nodiscard]] Eigen::VectorXcd Diagonal() const { return Eigen::VectorXcd::Ones(3); }
		[[nodiscard]] Eigen::VectorXcd ApplyOn(const Eigen::VectorXcd& vector) const { return vector; }
	};


	struct OperatorWithoutScalar
	{
		[[nodiscard]] Eigen::Index rows() const noexcept { return 3; }
		[[nodiscard]] Eigen::Index cols() const noexcept { return 3; }
		[[nodiscard]] Eigen::VectorXd Diagonal() const { return Eigen::VectorXd::Ones(3); }
		[[nodiscard]] Eigen::VectorXd ApplyOn(const Eigen::VectorXd& vector) const { return vector; }
	};
}


using namespace SecUtility::Math;


static_assert(SelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<double>>);
static_assert(SelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<std::complex<double>>>);
static_assert(BlockSelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<double>>);
static_assert(BlockSelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<std::complex<double>>>);
static_assert(!SelfAdjointLinearOperator<OperatorWithoutDiagonal>);
static_assert(!SelfAdjointLinearOperator<ComplexDiagonalOperator>);
static_assert(!SelfAdjointLinearOperator<OperatorWithoutScalar>);
static_assert(!SelfAdjointLinearOperator<DenseSelfAdjointLinearOperator<int>>);


TEMPLATE_TEST_CASE("iVI public result contract", "[Math][iVI]", double, (std::complex<double>))
{
	using Operator = DenseSelfAdjointLinearOperator<TestType>;
	using Solver = IterativeVectorInteractionSelfAdjointEigenSolver<Operator>;
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;

	Solver solver;
	STATIC_CHECK(std::is_same_v<typename Solver::Scalar, TestType>);
	STATIC_CHECK(std::is_same_v<typename Solver::RealScalar, RealScalar>);
	CHECK(solver.Status() == InteriorEigenSolverStatus::NotComputed);
	CHECK(solver.Eigenvalues().size() == 0);
	CHECK(solver.Eigenvectors().size() == 0);
	CHECK(solver.ResidualNorms().size() == 0);
	CHECK(solver.Statistics().CompletedIterationCount == 0);
	CHECK(solver.Statistics().MultipliedVectorCount == 0);
	CHECK(solver.Statistics().OperatorApplicationCount == 0);
	CHECK(solver.Statistics().MaximumExpansionSpaceSize == 0);
	CHECK(solver.Statistics().ExplicitImageRecalculationCount == 0);
	CHECK(solver.Statistics().GeneralizedSolveCount == 0);
}


TEMPLATE_TEST_CASE("iVI operator contract and input validation", "[Math][iVI]", double, (std::complex<double>))
{
	using RealScalar = typename Eigen::NumTraits<TestType>::Real;
	const auto operatorMatrix = IdentityOperator<TestType>(4);
	const EigenvalueInterval<RealScalar> interval{-1, 1};
	InteriorEigenSolverOptions<RealScalar> options;
	options.MaximumEigenpairCount = 2;

	SECTION("A valid problem is accepted")
	{
		CHECK_NOTHROW(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options));
		CHECK(interval.IsContaining(RealScalar{-1}));
		CHECK(interval.IsContaining(RealScalar{1}));
		CHECK_FALSE(interval.IsContaining(RealScalar{2}));
	}

	SECTION("A reversed interval is rejected")
	{
		CHECK_THROWS_AS(
		        ValidateInteriorEigenSolverInput(operatorMatrix, EigenvalueInterval<RealScalar>{1, -1}, options),
		        SecUtility::InvalidArgumentException);
	}

	SECTION("Non-finite interval bounds are rejected")
	{
		const auto infinity = std::numeric_limits<RealScalar>::infinity();
		CHECK_THROWS_AS(
		        ValidateInteriorEigenSolverInput(
		                operatorMatrix, EigenvalueInterval<RealScalar>{-infinity, 1}, options),
		        SecUtility::InvalidArgumentException);
	}

	SECTION("A non-positive eigenpair capacity is rejected")
	{
		options.MaximumEigenpairCount = 0;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);
	}

	SECTION("An eigenpair capacity larger than the dimension is rejected")
	{
		options.MaximumEigenpairCount = 5;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);
	}

	SECTION("Invalid iteration and retained-vector counts are rejected")
	{
		options.MaximumIterationCount = 0;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);

		options.MaximumIterationCount = 1;
		options.AdditionalRitzVectorCount = -1;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);
	}

	SECTION("Invalid numerical controls are rejected")
	{
		options.GeneralizedSolveInterval = 0;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);

		options.GeneralizedSolveInterval = 1;
		options.ResidualNormTolerance = 0;
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);

		options.ResidualNormTolerance = std::numeric_limits<RealScalar>::quiet_NaN();
		CHECK_THROWS_AS(ValidateInteriorEigenSolverInput(operatorMatrix, interval, options), SecUtility::InvalidArgumentException);
	}
}


TEST_CASE("iVI validation rejects inconsistent operator dimensions", "[Math][iVI]")
{
	InteriorEigenSolverOptions<double> options;
	options.MaximumEigenpairCount = 2;

	SECTION("The operator must be square")
	{
		const DenseSelfAdjointLinearOperator<double> operatorMatrix{Eigen::MatrixXd::Zero(3, 4)};
		CHECK_THROWS_AS(
		        ValidateInteriorEigenSolverInput(operatorMatrix, EigenvalueInterval<double>{-1, 1}, options),
		        SecUtility::InvalidArgumentException);
	}

	SECTION("The operator must have a positive dimension")
	{
		const DenseSelfAdjointLinearOperator<double> operatorMatrix{Eigen::MatrixXd::Zero(0, 0)};
		CHECK_THROWS_AS(
		        ValidateInteriorEigenSolverInput(operatorMatrix, EigenvalueInterval<double>{-1, 1}, options),
		        SecUtility::InvalidArgumentException);
	}

	SECTION("The diagonal must match the operator dimension")
	{
		struct IncorrectDiagonalOperator
		{
			using Scalar = double;
			[[nodiscard]] Eigen::Index rows() const noexcept { return 4; }
			[[nodiscard]] Eigen::Index cols() const noexcept { return 4; }
			[[nodiscard]] Eigen::VectorXd Diagonal() const { return Eigen::VectorXd::Ones(3); }
			[[nodiscard]] Eigen::VectorXd ApplyOn(const Eigen::VectorXd& vector) const { return vector; }
		};

		CHECK_THROWS_AS(
		        ValidateInteriorEigenSolverInput(
		                IncorrectDiagonalOperator{}, EigenvalueInterval<double>{-1, 1}, options),
		        SecUtility::InvalidArgumentException);
	}
}


TEST_CASE("The existing matrix-free wrapper requires a diagonal-aware adapter for iVI", "[Math][iVI]")
{
	const MatrixFreeLinearOperator operatorWithoutDiagonal([](const Eigen::VectorXd& vector) { return vector; }, 3);
	STATIC_CHECK_FALSE(SelfAdjointLinearOperator<decltype(operatorWithoutDiagonal)>);
}
