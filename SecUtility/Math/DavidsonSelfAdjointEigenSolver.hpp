// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if __cplusplus < 202002L
#error "DavidsonSelfAdjointEigenSolver requires C++20 or later"
#endif

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Math/SelfAdjointLinearOperator.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>


namespace SecUtility::Math
{
	template <typename RealScalar>
	struct DavidsonEigenSolverOptions
	{
		explicit constexpr DavidsonEigenSolverOptions(const Eigen::Index rootCount) noexcept
		    : RootCount(rootCount), InitialSubspaceDimension(rootCount)
		{
		}

		Eigen::Index RootCount;
		Eigen::Index MaximumIterationCount = 256;
		// The supplied basis may contain fewer columns; a later phase will provide explicit augmentation helpers.
		Eigen::Index InitialSubspaceDimension;
		// Zero selects the operator dimension.
		Eigen::Index MaximumSubspaceDimension = 0;
		RealScalar ResidualNormTolerance = static_cast<RealScalar>(1e-7);
		RealScalar EigenvalueChangeTolerance = static_cast<RealScalar>(1e-7);
		RealScalar PreconditionerDenominatorFloor = static_cast<RealScalar>(1e-12);
		RealScalar LinearDependenceTolerance = static_cast<RealScalar>(1e-10);
	};


	enum class DavidsonEigenSolverStatus
	{
		NotComputed,
		Converged,
		IterationLimitReached,
		ExpansionSpaceExhausted,
		NumericalFailure,
		StoppedByController
	};


	struct DavidsonEigenSolverStatistics
	{
		Eigen::Index CompletedIterationCount = 0;
		Eigen::Index OperatorApplicationCount = 0;
		Eigen::Index MultipliedVectorCount = 0;
		Eigen::Index MaximumSubspaceDimension = 0;
		Eigen::Index RestartCount = 0;
		Eigen::Index GeneratedCorrectionVectorCount = 0;
		Eigen::Index RetainedCorrectionVectorCount = 0;
		Eigen::Index OperatorRefreshCount = 0;
	};


	namespace Detail::Davidson
	{
		template <SelfAdjointLinearOperator Operator>
		void ValidateInput(
		        const Operator& linearOperator,
		        const Eigen::MatrixX<LinearOperatorScalar<Operator>>& initialBasis,
		        const DavidsonEigenSolverOptions<LinearOperatorRealScalar<Operator>>& options)
		{
			if (linearOperator.rows() != linearOperator.cols())
			{
				throw InvalidArgumentException("The self-adjoint linear operator must be square");
			}
			if (linearOperator.rows() <= 0)
			{
				throw InvalidArgumentException("The self-adjoint linear operator must have a positive dimension");
			}

			const Eigen::VectorX<LinearOperatorRealScalar<Operator>> diagonal = linearOperator.Diagonal();
			if (diagonal.size() != linearOperator.rows())
			{
				throw InvalidArgumentException("The linear operator diagonal has an inconsistent dimension");
			}
			if (!diagonal.allFinite())
			{
				throw InvalidArgumentException("The linear operator diagonal must contain only finite values");
			}
			if (initialBasis.rows() != linearOperator.rows())
			{
				throw InvalidArgumentException("The initial basis has an inconsistent row count");
			}
			if (initialBasis.cols() <= 0)
			{
				throw InvalidArgumentException("The initial basis must contain at least one vector");
			}
			if (initialBasis.cols() > linearOperator.rows())
			{
				throw InvalidArgumentException("The initial basis cannot contain more vectors than the operator dimension");
			}
			if (!initialBasis.allFinite())
			{
				throw InvalidArgumentException("The initial basis must contain only finite values");
			}
			if (options.RootCount <= 0 || options.RootCount > linearOperator.rows())
			{
				throw InvalidArgumentException("RootCount must be positive and not exceed the operator dimension");
			}
			if (options.MaximumIterationCount <= 0)
			{
				throw InvalidArgumentException("MaximumIterationCount must be positive");
			}
			if (options.InitialSubspaceDimension < options.RootCount
			    || options.InitialSubspaceDimension > linearOperator.rows())
			{
				throw InvalidArgumentException(
				        "InitialSubspaceDimension must accommodate every root and not exceed the operator dimension");
			}

			const Eigen::Index maximumSubspaceDimension = options.MaximumSubspaceDimension == 0
			                                                        ? linearOperator.rows()
			                                                        : options.MaximumSubspaceDimension;
			if (maximumSubspaceDimension < options.InitialSubspaceDimension
			    || maximumSubspaceDimension < initialBasis.cols()
			    || maximumSubspaceDimension > linearOperator.rows())
			{
				throw InvalidArgumentException(
				        "MaximumSubspaceDimension must accommodate the initial space and not exceed the operator dimension");
			}

			if (!std::isfinite(options.ResidualNormTolerance) || options.ResidualNormTolerance <= 0
			    || !std::isfinite(options.EigenvalueChangeTolerance) || options.EigenvalueChangeTolerance <= 0
			    || !std::isfinite(options.PreconditionerDenominatorFloor)
			    || options.PreconditionerDenominatorFloor <= 0
			    || !std::isfinite(options.LinearDependenceTolerance) || options.LinearDependenceTolerance <= 0)
			{
				throw InvalidArgumentException("All Davidson numerical tolerances must be finite and positive");
			}
		}
	}


	template <SelfAdjointLinearOperator Operator>
	class DavidsonSelfAdjointEigenSolver
	{
	public:
		using Scalar = LinearOperatorScalar<Operator>;
		using RealScalar = LinearOperatorRealScalar<Operator>;

		[[nodiscard]] DavidsonEigenSolverStatus Compute(
		        const Operator& linearOperator,
		        const Eigen::MatrixX<Scalar>& initialBasis,
		        const DavidsonEigenSolverOptions<RealScalar>& options)
		{
			Reset();
			Detail::Davidson::ValidateInput(linearOperator, initialBasis, options);
			ResetResultRows(linearOperator.rows());

			// Transitional Phase 2 terminal path. The first algorithmic result is implemented in Phase 4.
			m_Status = DavidsonEigenSolverStatus::IterationLimitReached;
			return m_Status;
		}

		[[nodiscard]] DavidsonEigenSolverStatus Status() const noexcept { return m_Status; }
		[[nodiscard]] const Eigen::VectorX<RealScalar>& Eigenvalues() const noexcept { return m_Eigenvalues; }
		[[nodiscard]] const Eigen::MatrixX<Scalar>& Eigenvectors() const noexcept { return m_Eigenvectors; }
		[[nodiscard]] const Eigen::VectorX<RealScalar>& ResidualNorms() const noexcept { return m_ResidualNorms; }
		[[nodiscard]] const DavidsonEigenSolverStatistics& Statistics() const noexcept { return m_Statistics; }
		[[nodiscard]] const Eigen::MatrixX<Scalar>& BasisVectors() const noexcept { return m_BasisVectors; }
		[[nodiscard]] const Eigen::MatrixX<Scalar>& BasisVectorImages() const noexcept { return m_BasisVectorImages; }
		[[nodiscard]] const Eigen::MatrixX<Scalar>& ReducedMatrix() const noexcept { return m_ReducedMatrix; }
		[[nodiscard]] const Eigen::MatrixX<Scalar>& ReducedEigenvectors() const noexcept
		{
			return m_ReducedEigenvectors;
		}

	private:
		void Reset()
		{
			m_Status = DavidsonEigenSolverStatus::NotComputed;
			m_Eigenvalues.resize(0);
			m_Eigenvectors.resize(0, 0);
			m_ResidualNorms.resize(0);
			m_Statistics = {};
			m_BasisVectors.resize(0, 0);
			m_BasisVectorImages.resize(0, 0);
			m_ReducedMatrix.resize(0, 0);
			m_ReducedEigenvectors.resize(0, 0);
		}

		void ResetResultRows(const Eigen::Index dimension)
		{
			m_Eigenvectors.resize(dimension, 0);
			m_BasisVectors.resize(dimension, 0);
			m_BasisVectorImages.resize(dimension, 0);
		}

		DavidsonEigenSolverStatus m_Status = DavidsonEigenSolverStatus::NotComputed;
		Eigen::VectorX<RealScalar> m_Eigenvalues;
		Eigen::MatrixX<Scalar> m_Eigenvectors;
		Eigen::VectorX<RealScalar> m_ResidualNorms;
		DavidsonEigenSolverStatistics m_Statistics;
		Eigen::MatrixX<Scalar> m_BasisVectors;
		Eigen::MatrixX<Scalar> m_BasisVectorImages;
		Eigen::MatrixX<Scalar> m_ReducedMatrix;
		Eigen::MatrixX<Scalar> m_ReducedEigenvectors;
	};
}
