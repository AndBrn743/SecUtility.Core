// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if __cplusplus < 202002L
#error "IterativeVectorInteractionSelfAdjointEigenSolver requires C++20 or later"
#endif

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Math/Core.hpp>

#include <Eigen/Core>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <type_traits>


namespace SecUtility::Math
{
	template <typename RealScalar>
	struct EigenvalueInterval
	{
		RealScalar LowerBound{};
		RealScalar UpperBound{};

		[[nodiscard]] constexpr bool IsValid() const noexcept
		{
			return LowerBound <= UpperBound;
		}

		[[nodiscard]] constexpr bool IsContaining(const RealScalar value) const noexcept
		{
			return value >= LowerBound && value <= UpperBound;
		}
	};


	template <typename RealScalar>
	struct InteriorEigenSolverOptions
	{
		Eigen::Index MaximumEigenpairCount = 0;
		Eigen::Index MaximumIterationCount = 1000;
		// Zero selects the automatic policy: retain as many additional Ritz vectors as current primary vectors.
		Eigen::Index AdditionalRitzVectorCount = 0;
		Eigen::Index GeneralizedSolveInterval = 5;

		RealScalar EigenvalueChangeTolerance = static_cast<RealScalar>(1e-7);
		RealScalar ResidualNormTolerance = static_cast<RealScalar>(1e-7);
		RealScalar PreconditionerDenominatorFloor = static_cast<RealScalar>(1e-2);
		RealScalar LinearDependenceTolerance = static_cast<RealScalar>(1e-10);

		bool IsPreviousRitzVectorRecyclingEnabled = true;
		bool IsFreezingEnabled = true;
	};


	enum class InteriorEigenSolverStatus
	{
		NotComputed,
		Converged,
		NoEigenpairsFound,
		MaximumEigenpairCountExceeded,
		IterationLimitReached,
		ExpansionSpaceExhausted,
		NumericalFailure
	};


	struct InteriorEigenSolverStatistics
	{
		Eigen::Index CompletedIterationCount = 0;
		Eigen::Index MultipliedVectorCount = 0;
		Eigen::Index OperatorApplicationCount = 0;
		Eigen::Index MaximumExpansionSpaceSize = 0;
		Eigen::Index ExplicitImageRecalculationCount = 0;
		Eigen::Index GeneralizedSolveCount = 0;
	};


	template <typename Operator>
	using LinearOperatorScalar = std::remove_cvref_t<Operator>::Scalar;

	template <typename Operator>
	using LinearOperatorRealScalar = Eigen::NumTraits<LinearOperatorScalar<Operator>>::Real;


	template <typename Operator>
	concept SelfAdjointLinearOperator = requires(
	        const std::remove_reference_t<Operator>& linearOperator,
	        const Eigen::VectorX<LinearOperatorScalar<Operator>>& vector)
	{
		typename LinearOperatorScalar<Operator>;
		typename LinearOperatorRealScalar<Operator>;
		requires !Eigen::NumTraits<LinearOperatorScalar<Operator>>::IsInteger;

		{ linearOperator.rows() } -> std::convertible_to<Eigen::Index>;
		{ linearOperator.cols() } -> std::convertible_to<Eigen::Index>;

		{ linearOperator.Diagonal().size() } -> std::convertible_to<Eigen::Index>;
		{ linearOperator.Diagonal()[Eigen::Index{}] } -> std::convertible_to<LinearOperatorRealScalar<Operator>>;

		{ linearOperator.ApplyOn(vector).size() } -> std::convertible_to<Eigen::Index>;
		{ linearOperator.ApplyOn(vector)[Eigen::Index{}] } -> std::convertible_to<LinearOperatorScalar<Operator>>;
	};


	template <typename Operator>
	concept BlockSelfAdjointLinearOperator = SelfAdjointLinearOperator<Operator>
	                                      && requires(
	                                              const std::remove_reference_t<Operator>& linearOperator,
	                                              const Eigen::MatrixX<LinearOperatorScalar<Operator>>& vectors)
	{
		{ linearOperator.ApplyOn(vectors).rows() } -> std::convertible_to<Eigen::Index>;
		{ linearOperator.ApplyOn(vectors).cols() } -> std::convertible_to<Eigen::Index>;
		{ linearOperator.ApplyOn(vectors)(Eigen::Index{}, Eigen::Index{}) }
		        -> std::convertible_to<LinearOperatorScalar<Operator>>;
	};


	template <SelfAdjointLinearOperator Operator>
	class IterativeVectorInteractionSelfAdjointEigenSolver
	{
	public:
		using Scalar = LinearOperatorScalar<Operator>;
		using RealScalar = LinearOperatorRealScalar<Operator>;

		void Compute(const Operator& linearOperator,
		             const EigenvalueInterval<RealScalar>& interval,
		             const InteriorEigenSolverOptions<RealScalar>& options);

		[[nodiscard]] InteriorEigenSolverStatus Status() const noexcept
		{
			return m_Status;
		}

		[[nodiscard]] const Eigen::VectorX<RealScalar>& Eigenvalues() const noexcept
		{
			return m_Eigenvalues;
		}

		[[nodiscard]] const Eigen::MatrixX<Scalar>& Eigenvectors() const noexcept
		{
			return m_Eigenvectors;
		}

		[[nodiscard]] const Eigen::VectorX<RealScalar>& ResidualNorms() const noexcept
		{
			return m_ResidualNorms;
		}

		[[nodiscard]] const InteriorEigenSolverStatistics& Statistics() const noexcept
		{
			return m_Statistics;
		}

	private:
		InteriorEigenSolverStatus m_Status = InteriorEigenSolverStatus::NotComputed;
		Eigen::VectorX<RealScalar> m_Eigenvalues;
		Eigen::MatrixX<Scalar> m_Eigenvectors;
		Eigen::VectorX<RealScalar> m_ResidualNorms;
		InteriorEigenSolverStatistics m_Statistics;
	};


	template <SelfAdjointLinearOperator Operator>
	void ValidateInteriorEigenSolverInput(
	        const Operator& linearOperator,
	        const EigenvalueInterval<LinearOperatorRealScalar<Operator>>& interval,
	        const InteriorEigenSolverOptions<LinearOperatorRealScalar<Operator>>& options)
	{
		if (linearOperator.rows() != linearOperator.cols())
		{
			throw InvalidArgumentException("The self-adjoint linear operator must be square");
		}
		if (linearOperator.rows() <= 0)
		{
			throw InvalidArgumentException("The self-adjoint linear operator must have a positive dimension");
		}
		if (linearOperator.Diagonal().size() != linearOperator.rows())
		{
			throw InvalidArgumentException("The linear operator diagonal has an inconsistent dimension");
		}
		if (!interval.IsValid())
		{
			throw InvalidArgumentException("The eigenvalue interval lower bound must not exceed its upper bound");
		}
		if (!std::isfinite(interval.LowerBound) || !std::isfinite(interval.UpperBound))
		{
			throw InvalidArgumentException("The eigenvalue interval bounds must be finite");
		}
		if (options.MaximumEigenpairCount <= 0)
		{
			throw InvalidArgumentException("MaximumEigenpairCount must be positive");
		}
		if (options.MaximumEigenpairCount > linearOperator.rows())
		{
			throw InvalidArgumentException("MaximumEigenpairCount must not exceed the operator dimension");
		}
		if (options.MaximumIterationCount <= 0)
		{
			throw InvalidArgumentException("MaximumIterationCount must be positive");
		}
		if (options.AdditionalRitzVectorCount < 0)
		{
			throw InvalidArgumentException("AdditionalRitzVectorCount must not be negative");
		}
		if (options.GeneralizedSolveInterval <= 0)
		{
			throw InvalidArgumentException("GeneralizedSolveInterval must be positive");
		}
		if (!std::isfinite(options.EigenvalueChangeTolerance) || options.EigenvalueChangeTolerance <= 0
		    || !std::isfinite(options.ResidualNormTolerance) || options.ResidualNormTolerance <= 0
		    || !std::isfinite(options.PreconditionerDenominatorFloor)
		    || options.PreconditionerDenominatorFloor <= 0 || !std::isfinite(options.LinearDependenceTolerance)
		    || options.LinearDependenceTolerance <= 0)
		{
			throw InvalidArgumentException("All numerical tolerances must be positive");
		}
	}
}  // namespace SecUtility::Math
