// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if __cplusplus < 202002L
#error "IterativeVectorInteractionSelfAdjointEigenSolver requires C++20 or later"
#endif

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Math/Core.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
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


	namespace Detail::IterativeVectorInteraction
	{
		template <typename Scalar>
		struct VectorImagePair
		{
			Eigen::MatrixX<Scalar> Vectors;
			Eigen::MatrixX<Scalar> Images;
		};
	}


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


	namespace Detail::IterativeVectorInteraction
	{
		template <SelfAdjointLinearOperator Operator>
		Eigen::MatrixX<LinearOperatorScalar<Operator>> ApplyOperator(
		        const Operator& linearOperator,
		        const Eigen::MatrixX<LinearOperatorScalar<Operator>>& vectors,
		        InteriorEigenSolverStatistics& ref_statistics)
		{
			using Scalar = LinearOperatorScalar<Operator>;
			Eigen::MatrixX<Scalar> images(linearOperator.rows(), vectors.cols());

			if (vectors.cols() == 0)
			{
				return images;
			}

			if constexpr (BlockSelfAdjointLinearOperator<Operator>)
			{
				images = linearOperator.ApplyOn(vectors);
				ref_statistics.OperatorApplicationCount++;
			}
			else
			{
				for (Eigen::Index columnIndex = 0; columnIndex < vectors.cols(); columnIndex++)
				{
					images.col(columnIndex) = linearOperator.ApplyOn(vectors.col(columnIndex));
					ref_statistics.OperatorApplicationCount++;
				}
			}

			ref_statistics.MultipliedVectorCount += vectors.cols();
			return images;
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> ProjectAgainstBasis(const Eigen::MatrixX<Scalar>& basis,
		                                           const Eigen::MatrixX<Scalar>& candidates)
		{
			if (basis.cols() == 0)
			{
				return candidates;
			}
			return candidates - basis * (basis.adjoint() * candidates);
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> SymmetricallyOrthonormalize(
		        const Eigen::MatrixX<Scalar>& vectors,
		        const typename Eigen::NumTraits<Scalar>::Real relativeLinearDependenceTolerance)
		{
			using RealScalar = Eigen::NumTraits<Scalar>::Real;
			if (vectors.cols() == 0)
			{
				return vectors;
			}

			const Eigen::MatrixX<Scalar> gramMatrix = vectors.adjoint() * vectors;
			const Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> eigenSolver(gramMatrix);
			if (eigenSolver.info() != Eigen::Success)
			{
				throw OperationFailedException("Symmetric orthonormalization failed to diagonalize the Gram matrix");
			}

			const auto& gramEigenvalues = eigenSolver.eigenvalues();
			const RealScalar largestEigenvalue = gramEigenvalues.cend()[-1];
			const RealScalar rejectionThreshold =
			        relativeLinearDependenceTolerance * Max(RealScalar{1}, largestEigenvalue);
			const auto firstAcceptedIndex = static_cast<Eigen::Index>(std::distance(
			        gramEigenvalues.begin(),
			        std::find_if(gramEigenvalues.begin(),
			                     gramEigenvalues.end(),
			                     [rejectionThreshold](const RealScalar eigenvalue)
			                     { return eigenvalue >= rejectionThreshold; })));
			const Eigen::Index acceptedCount = vectors.cols() - firstAcceptedIndex;
			if (acceptedCount == 0)
			{
				return Eigen::MatrixX<Scalar>(vectors.rows(), 0);
			}

			const auto acceptedEigenvalues = gramEigenvalues.tail(acceptedCount);
			const auto acceptedEigenvectors = eigenSolver.eigenvectors().rightCols(acceptedCount);
			return vectors * acceptedEigenvectors * acceptedEigenvalues.cwiseInverse().cwiseSqrt().asDiagonal();
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> OrthogonalizeAndRemoveLinearDependence(
		        const Eigen::MatrixX<Scalar>& basis,
		        const Eigen::MatrixX<Scalar>& candidates,
		        const typename Eigen::NumTraits<Scalar>::Real relativeLinearDependenceTolerance)
		{
			return SymmetricallyOrthonormalize(
			        ProjectAgainstBasis(basis, candidates), relativeLinearDependenceTolerance);
		}


		template <typename Scalar>
		VectorImagePair<Scalar> TransformVectorImagePair(const VectorImagePair<Scalar>& source,
		                                                 const Eigen::MatrixX<Scalar>& coefficients)
		{
			return {source.Vectors * coefficients, source.Images * coefficients};
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> FormReducedMatrix(const VectorImagePair<Scalar>& vectorImagePair)
		{
			return vectorImagePair.Vectors.adjoint() * vectorImagePair.Images;
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> CalculateResiduals(
		        const VectorImagePair<Scalar>& vectorImagePair,
		        const Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& eigenvalues)
		{
			return vectorImagePair.Images - vectorImagePair.Vectors * eigenvalues.asDiagonal();
		}


		template <typename Scalar>
		Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real> CalculateColumnNorms(
		        const Eigen::MatrixX<Scalar>& vectors)
		{
			using RealScalar = Eigen::NumTraits<Scalar>::Real;
			Eigen::VectorX<RealScalar> norms(vectors.cols());
			for (Eigen::Index columnIndex = 0; columnIndex < vectors.cols(); columnIndex++)
			{
				norms[columnIndex] = vectors.col(columnIndex).norm();
			}
			return norms;
		}


		template <typename Scalar>
		Eigen::MatrixX<Scalar> ApplyAbsoluteDiagonalPreconditioner(
		        const Eigen::MatrixX<Scalar>& residuals,
		        const Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& eigenvalues,
		        const Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& diagonal,
		        const typename Eigen::NumTraits<Scalar>::Real denominatorFloor)
		{
			using RealScalar = Eigen::NumTraits<Scalar>::Real;
			Eigen::MatrixX<Scalar> corrections(residuals.rows(), residuals.cols());
			for (Eigen::Index columnIndex = 0; columnIndex < residuals.cols(); columnIndex++)
			{
				for (Eigen::Index rowIndex = 0; rowIndex < residuals.rows(); rowIndex++)
				{
					const RealScalar denominator =
					        Max(Abs(diagonal[rowIndex] - eigenvalues[columnIndex]), denominatorFloor);
					corrections(rowIndex, columnIndex) = residuals(rowIndex, columnIndex) / denominator;
				}
			}
			return corrections;
		}
	}


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
