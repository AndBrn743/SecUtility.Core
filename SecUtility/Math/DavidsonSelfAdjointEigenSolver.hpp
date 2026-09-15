// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if __cplusplus < 202002L
#error "DavidsonSelfAdjointEigenSolver requires C++20 or later"
#endif

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Math/SelfAdjointLinearOperator.hpp>
#include <SecUtility/Raw/Int.hpp>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>


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
		Eigen::Index AdditionalRestartRitzVectorCount = 0;
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
		template <typename Scalar>
		struct VectorImageSubspace
		{
			Eigen::MatrixX<Scalar> Vectors;
			Eigen::MatrixX<Scalar> Images;
			Eigen::MatrixX<Scalar> ReducedMatrix;
		};


		template <typename Scalar>
		struct RitzAnalysis
		{
			Eigen::ComputationInfo ComputationInfo = Eigen::Success;
			Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real> AllEigenvalues;
			Eigen::MatrixX<Scalar> AllReducedEigenvectors;
			Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real> Eigenvalues;
			Eigen::MatrixX<Scalar> Eigenvectors;
			Eigen::MatrixX<Scalar> EigenvectorImages;
			Eigen::MatrixX<Scalar> Residuals;
			Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real> ResidualNorms;
			Eigen::MatrixX<Scalar> ReducedEigenvectors;
		};


		template <typename Scalar>
		struct DiagonalCorrectionContext
		{
			const Eigen::MatrixX<Scalar>& Residuals;
			const Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& RitzValues;
			const Eigen::VectorX<typename Eigen::NumTraits<Scalar>::Real>& OperatorDiagonal;
			const Eigen::ArrayX<Int8>& RootConvergenceIndicators;
			const Eigen::NumTraits<Scalar>::Real DenominatorFloor;
		};


		template <typename Scalar>
		struct CorrectionCandidates
		{
			Eigen::MatrixX<Scalar> Vectors;
			std::vector<Eigen::Index> SourceRootIndices;
		};


		template <typename Scalar>
		Eigen::MatrixX<Scalar> OrthonormalizeAndRemoveLinearDependence(
		        const Eigen::MatrixX<Scalar>& vectors,
		        const typename Eigen::NumTraits<Scalar>::Real relativeLinearDependenceTolerance)
		{
			Eigen::ColPivHouseholderQR<Eigen::MatrixX<Scalar>> qr(vectors);
			qr.setThreshold(relativeLinearDependenceTolerance);
			const Eigen::Index rank = qr.rank();
			if (rank == 0)
			{
				return Eigen::MatrixX<Scalar>(vectors.rows(), 0);
			}
			return qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(vectors.rows(), rank);
		}


		template <typename RealScalar>
		RealScalar RegularizeSignedDenominator(const RealScalar denominator, const RealScalar denominatorFloor)
		{
			assert(denominatorFloor > RealScalar{0});
			if (Abs(denominator) >= denominatorFloor)
			{
				return denominator;
			}
			return CopySignToTheLeft(denominatorFloor, denominator);
		}


		template <typename Scalar>
		CorrectionCandidates<Scalar> GenerateDiagonalCorrectionCandidates(
		        const DiagonalCorrectionContext<Scalar>& context, DavidsonEigenSolverStatistics& ref_statistics)
		{
			assert(context.Residuals.cols() == context.RitzValues.size());
			assert(context.Residuals.rows() == context.OperatorDiagonal.size());
			assert(context.RootConvergenceIndicators.size() == context.RitzValues.size());
			const Eigen::Index unconvergedRootCount =
			        (context.RootConvergenceIndicators == 0).template cast<Eigen::Index>().sum();
			CorrectionCandidates<Scalar> candidates{
			        Eigen::MatrixX<Scalar>(context.Residuals.rows(), unconvergedRootCount), {}};
			candidates.SourceRootIndices.reserve(static_cast<std::size_t>(unconvergedRootCount));
			Eigen::Index candidateIndex = 0;
			for (Eigen::Index rootIndex = 0; rootIndex < context.RitzValues.size(); rootIndex++)
			{
				if (context.RootConvergenceIndicators[rootIndex] != 0)
				{
					continue;
				}
				for (Eigen::Index rowIndex = 0; rowIndex < context.Residuals.rows(); rowIndex++)
				{
					const auto denominator = RegularizeSignedDenominator(context.RitzValues[rootIndex]
					                                                             - context.OperatorDiagonal[rowIndex],
					                                                     context.DenominatorFloor);
					// t = -(D - theta I)^-1 r = r / (theta - D).
					candidates.Vectors(rowIndex, candidateIndex) = context.Residuals(rowIndex, rootIndex) / denominator;
				}
				candidates.SourceRootIndices.push_back(rootIndex);
				candidateIndex++;
			}
			ref_statistics.GeneratedCorrectionVectorCount += unconvergedRootCount;
			return candidates;
		}


		template <typename Scalar>
		CorrectionCandidates<Scalar> OrthogonalizeCorrectionCandidates(
		        const Eigen::MatrixX<Scalar>& basis,
		        const CorrectionCandidates<Scalar>& candidates,
		        const typename Eigen::NumTraits<Scalar>::Real relativeLinearDependenceTolerance,
		        DavidsonEigenSolverStatistics& ref_statistics)
		{
			if (candidates.Vectors.cols() == 0)
			{
				return {Eigen::MatrixX<Scalar>(basis.rows(), 0), {}};
			}

			Eigen::MatrixX<Scalar> projected = candidates.Vectors - basis * (basis.adjoint() * candidates.Vectors);
			projected -= basis * (basis.adjoint() * projected);
			Eigen::ColPivHouseholderQR<Eigen::MatrixX<Scalar>> qr(projected);
			qr.setThreshold(relativeLinearDependenceTolerance);
			const Eigen::Index rank = qr.rank();
			if (rank == 0)
			{
				return {Eigen::MatrixX<Scalar>(basis.rows(), 0), {}};
			}

			CorrectionCandidates<Scalar> orthonormalized{
			        qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(projected.rows(), rank), {}};
			orthonormalized.SourceRootIndices.reserve(static_cast<std::size_t>(rank));
			for (Eigen::Index pivotIndex = 0; pivotIndex < rank; pivotIndex++)
			{
				const Eigen::Index sourceCandidateIndex = qr.colsPermutation().indices()[pivotIndex];
				orthonormalized.SourceRootIndices.push_back(
				        candidates.SourceRootIndices[static_cast<std::size_t>(sourceCandidateIndex)]);
			}
			ref_statistics.RetainedCorrectionVectorCount += rank;
			return orthonormalized;
		}


		template <SelfAdjointLinearOperator Operator>
		VectorImageSubspace<LinearOperatorScalar<Operator>> CreateInitialSubspace(
		        const Operator& linearOperator,
		        const Eigen::MatrixX<LinearOperatorScalar<Operator>>& initialBasis,
		        const LinearOperatorRealScalar<Operator> relativeLinearDependenceTolerance,
		        DavidsonEigenSolverStatistics& ref_statistics)
		{
			using Scalar = LinearOperatorScalar<Operator>;
			Eigen::MatrixX<Scalar> vectors =
			        OrthonormalizeAndRemoveLinearDependence(initialBasis, relativeLinearDependenceTolerance);
			if (vectors.cols() == 0)
			{
				throw InvalidArgumentException("The initial basis does not contain a linearly independent vector");
			}

			Eigen::MatrixX<Scalar> images = ApplySelfAdjointLinearOperator(linearOperator, vectors);
			ref_statistics.OperatorApplicationCount += BlockSelfAdjointLinearOperator<Operator> ? 1 : vectors.cols();
			ref_statistics.MultipliedVectorCount += vectors.cols();
			ref_statistics.MaximumSubspaceDimension = vectors.cols();
			Eigen::MatrixX<Scalar> reducedMatrix = vectors.adjoint() * images;
			return {std::move(vectors), std::move(images), std::move(reducedMatrix)};
		}


		template <typename Scalar>
		RitzAnalysis<Scalar> AnalyzeSubspace(const VectorImageSubspace<Scalar>& subspace,
		                                     const Eigen::Index requestedRootCount)
		{
			RitzAnalysis<Scalar> analysis;
			if (!subspace.ReducedMatrix.allFinite())
			{
				analysis.ComputationInfo = Eigen::NumericalIssue;
				return analysis;
			}

			Eigen::SelfAdjointEigenSolver<Eigen::MatrixX<Scalar>> reducedSolver(subspace.ReducedMatrix);
			analysis.ComputationInfo = reducedSolver.info();
			if (analysis.ComputationInfo != Eigen::Success)
			{
				return analysis;
			}

			const Eigen::Index rootCount = Min(requestedRootCount, subspace.Vectors.cols());
			analysis.AllEigenvalues = reducedSolver.eigenvalues();
			analysis.AllReducedEigenvectors = reducedSolver.eigenvectors();
			analysis.Eigenvalues = analysis.AllEigenvalues.head(rootCount);
			analysis.ReducedEigenvectors = analysis.AllReducedEigenvectors.leftCols(rootCount);
			analysis.Eigenvectors = subspace.Vectors * analysis.ReducedEigenvectors;
			analysis.EigenvectorImages = subspace.Images * analysis.ReducedEigenvectors;
			analysis.Residuals = analysis.EigenvectorImages
			                     - analysis.Eigenvectors * analysis.Eigenvalues.template cast<Scalar>().asDiagonal();
			analysis.ResidualNorms = analysis.Residuals.colwise().norm().transpose();
			return analysis;
		}


		template <typename Scalar>
		VectorImageSubspace<Scalar> RestartSubspace(const VectorImageSubspace<Scalar>& subspace,
		                                            const RitzAnalysis<Scalar>& analysis,
		                                            const Eigen::Index requestedRootCount,
		                                            const Eigen::Index additionalRitzVectorCount,
		                                            const Eigen::Index maximumSubspaceDimension)
		{
			assert(additionalRitzVectorCount >= 0);
			assert(maximumSubspaceDimension > 0);
			const Eigen::Index requiredRitzVectorCount =
			        Min(requestedRootCount, analysis.AllReducedEigenvectors.cols());
			const Eigen::Index preferredRitzVectorCount =
			        Min(analysis.AllReducedEigenvectors.cols(), requiredRitzVectorCount + additionalRitzVectorCount);
			const Eigen::Index maximumRetainedCount = maximumSubspaceDimension > requiredRitzVectorCount
			                                                  ? maximumSubspaceDimension - 1
			                                                  : requiredRitzVectorCount;
			const Eigen::Index retainedCount = Min(preferredRitzVectorCount, maximumRetainedCount);
			const Eigen::MatrixX<Scalar> coefficients = analysis.AllReducedEigenvectors.leftCols(retainedCount);
			VectorImageSubspace<Scalar> restarted{subspace.Vectors * coefficients, subspace.Images * coefficients, {}};
			restarted.ReducedMatrix = restarted.Vectors.adjoint() * restarted.Images;
			return restarted;
		}


		template <SelfAdjointLinearOperator Operator>
		Eigen::MatrixX<LinearOperatorScalar<Operator>> ApplyOperator(
		        const Operator& linearOperator,
		        const Eigen::MatrixX<LinearOperatorScalar<Operator>>& vectors,
		        DavidsonEigenSolverStatistics& ref_statistics)
		{
			const auto images = ApplySelfAdjointLinearOperator(linearOperator, vectors);
			if (vectors.cols() > 0)
			{
				ref_statistics.OperatorApplicationCount +=
				        BlockSelfAdjointLinearOperator<Operator> ? 1 : vectors.cols();
				ref_statistics.MultipliedVectorCount += vectors.cols();
			}
			return images;
		}


		template <SelfAdjointLinearOperator Operator>
		void ValidateInput(const Operator& linearOperator,
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
				throw InvalidArgumentException(
				        "The initial basis cannot contain more vectors than the operator dimension");
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
			if (options.AdditionalRestartRitzVectorCount < 0)
			{
				throw InvalidArgumentException("AdditionalRestartRitzVectorCount must not be negative");
			}
			if (options.InitialSubspaceDimension < options.RootCount
			    || options.InitialSubspaceDimension > linearOperator.rows())
			{
				throw InvalidArgumentException(
				        "InitialSubspaceDimension must accommodate every root and not exceed the operator dimension");
			}

			const Eigen::Index maximumSubspaceDimension =
			        options.MaximumSubspaceDimension == 0 ? linearOperator.rows() : options.MaximumSubspaceDimension;
			if (maximumSubspaceDimension < options.InitialSubspaceDimension
			    || maximumSubspaceDimension < initialBasis.cols() || maximumSubspaceDimension > linearOperator.rows())
			{
				throw InvalidArgumentException("MaximumSubspaceDimension must accommodate the initial space and not "
				                               "exceed the operator dimension");
			}

			if (!std::isfinite(options.ResidualNormTolerance) || options.ResidualNormTolerance <= 0
			    || !std::isfinite(options.EigenvalueChangeTolerance) || options.EigenvalueChangeTolerance <= 0
			    || !std::isfinite(options.PreconditionerDenominatorFloor) || options.PreconditionerDenominatorFloor <= 0
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

		[[nodiscard]] DavidsonEigenSolverStatus Compute(const Operator& linearOperator,
		                                                const Eigen::MatrixX<Scalar>& initialBasis,
		                                                const DavidsonEigenSolverOptions<RealScalar>& options)
		{
			Reset();
			Detail::Davidson::ValidateInput(linearOperator, initialBasis, options);
			ResetResultRows(linearOperator.rows());
			auto subspace = Detail::Davidson::CreateInitialSubspace(
			        linearOperator, initialBasis, options.LinearDependenceTolerance, m_Statistics);
			const Eigen::VectorX<RealScalar> diagonal = linearOperator.Diagonal();
			const Eigen::Index maximumSubspaceDimension =
			        options.MaximumSubspaceDimension == 0 ? linearOperator.rows() : options.MaximumSubspaceDimension;

			for (Eigen::Index iterationIndex = 0; iterationIndex < options.MaximumIterationCount; iterationIndex++)
			{
				m_Statistics.CompletedIterationCount = iterationIndex + 1;
				const auto analysis = Detail::Davidson::AnalyzeSubspace(subspace, options.RootCount);
				if (analysis.ComputationInfo != Eigen::Success)
				{
					StoreSubspace(std::move(subspace));
					m_Status = DavidsonEigenSolverStatus::NumericalFailure;
					return m_Status;
				}

				PublishRitzResults(analysis);
				const bool hasEveryRequestedRoot = m_Eigenvalues.size() == options.RootCount;
				const Eigen::ArrayX<Int8> rootConvergenceIndicators =
				        (m_ResidualNorms.array() <= options.ResidualNormTolerance).template cast<Int8>();
				if (hasEveryRequestedRoot && (rootConvergenceIndicators != 0).all())
				{
					StoreSubspace(std::move(subspace));
					m_Status = DavidsonEigenSolverStatus::Converged;
					return m_Status;
				}
				if (iterationIndex + 1 == options.MaximumIterationCount)
				{
					StoreSubspace(std::move(subspace));
					m_Status = DavidsonEigenSolverStatus::IterationLimitReached;
					return m_Status;
				}

				const auto generatedCorrections = Detail::Davidson::GenerateDiagonalCorrectionCandidates(
				        Detail::Davidson::DiagonalCorrectionContext{analysis.Residuals,
				                                                    analysis.Eigenvalues,
				                                                    diagonal,
				                                                    rootConvergenceIndicators,
				                                                    options.PreconditionerDenominatorFloor},
				        m_Statistics);
				const auto corrections = Detail::Davidson::OrthogonalizeCorrectionCandidates(
				        subspace.Vectors, generatedCorrections, options.LinearDependenceTolerance, m_Statistics);
				if (corrections.Vectors.cols() == 0)
				{
					StoreSubspace(std::move(subspace));
					m_Status = DavidsonEigenSolverStatus::ExpansionSpaceExhausted;
					return m_Status;
				}
				if (subspace.Vectors.cols() + corrections.Vectors.cols() > maximumSubspaceDimension)
				{
					subspace = Detail::Davidson::RestartSubspace(subspace,
					                                             analysis,
					                                             options.RootCount,
					                                             options.AdditionalRestartRitzVectorCount,
					                                             maximumSubspaceDimension);
					m_Statistics.RestartCount++;
				}

				const Eigen::Index availableCorrectionCount = maximumSubspaceDimension - subspace.Vectors.cols();
				if (availableCorrectionCount <= 0)
				{
					StoreSubspace(std::move(subspace));
					m_Status = DavidsonEigenSolverStatus::ExpansionSpaceExhausted;
					return m_Status;
				}
				const Eigen::Index appendedCorrectionCount = Min(availableCorrectionCount, corrections.Vectors.cols());
				const Eigen::MatrixX<Scalar> appendedCorrections =
				        corrections.Vectors.leftCols(appendedCorrectionCount);

				const Eigen::MatrixX<Scalar> correctionImages =
				        Detail::Davidson::ApplyOperator(linearOperator, appendedCorrections, m_Statistics);
				const Eigen::Index oldSubspaceDimension = subspace.Vectors.cols();
				subspace.Vectors.conservativeResize(Eigen::NoChange, oldSubspaceDimension + appendedCorrectionCount);
				subspace.Images.conservativeResize(Eigen::NoChange, oldSubspaceDimension + appendedCorrectionCount);
				subspace.Vectors.rightCols(appendedCorrectionCount) = appendedCorrections;
				subspace.Images.rightCols(appendedCorrectionCount) = correctionImages;
				subspace.ReducedMatrix = subspace.Vectors.adjoint() * subspace.Images;
				m_Statistics.MaximumSubspaceDimension =
				        Max(m_Statistics.MaximumSubspaceDimension, subspace.Vectors.cols());
			}

			throw InvariantViolationException("Davidson iteration terminated without a status");
		}

		[[nodiscard]] DavidsonEigenSolverStatus Status() const noexcept
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

		[[nodiscard]] const DavidsonEigenSolverStatistics& Statistics() const noexcept
		{
			return m_Statistics;
		}

		[[nodiscard]] const Eigen::MatrixX<Scalar>& BasisVectors() const noexcept
		{
			return m_BasisVectors;
		}

		[[nodiscard]] const Eigen::MatrixX<Scalar>& BasisVectorImages() const noexcept
		{
			return m_BasisVectorImages;
		}

		[[nodiscard]] const Eigen::MatrixX<Scalar>& ReducedMatrix() const noexcept
		{
			return m_ReducedMatrix;
		}

		[[nodiscard]] const Eigen::MatrixX<Scalar>& ReducedEigenvectors() const noexcept
		{
			return m_ReducedEigenvectors;
		}

	private:
		void PublishRitzResults(const Detail::Davidson::RitzAnalysis<Scalar>& analysis)
		{
			m_Eigenvalues = analysis.Eigenvalues;
			m_Eigenvectors = analysis.Eigenvectors;
			m_ResidualNorms = analysis.ResidualNorms;
			m_ReducedEigenvectors = analysis.ReducedEigenvectors;
		}

		void StoreSubspace(Detail::Davidson::VectorImageSubspace<Scalar>&& subspace)
		{
			m_BasisVectors = std::move(subspace.Vectors);
			m_BasisVectorImages = std::move(subspace.Images);
			m_ReducedMatrix = std::move(subspace.ReducedMatrix);
		}

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
