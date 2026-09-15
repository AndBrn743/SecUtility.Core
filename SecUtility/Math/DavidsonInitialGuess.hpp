// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>

#include <Eigen/Core>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>


namespace SecUtility::Math
{
	namespace Detail::DavidsonInitialGuess
	{
		inline void ValidateDimensions(const Eigen::Index dimension, const Eigen::Index vectorCount)
		{
			if (dimension <= 0 || vectorCount <= 0 || vectorCount > dimension)
			{
				throw InvalidArgumentException("A Davidson initial basis requires 0 < vectorCount <= dimension");
			}
		}


		template <typename Scalar>
		Scalar RandomScalar(std::mt19937_64& generator, std::normal_distribution<double>& distribution)
		{
			using RealScalar = Eigen::NumTraits<Scalar>::Real;
			if constexpr (Eigen::NumTraits<Scalar>::IsComplex)
			{
				return Scalar{static_cast<RealScalar>(distribution(generator)),
				              static_cast<RealScalar>(distribution(generator))};
			}
			else
			{
				return static_cast<Scalar>(distribution(generator));
			}
		}
	}


	template <typename Scalar>
	[[nodiscard]] Eigen::MatrixX<Scalar> IdentityPrefixDavidsonInitialBasis(
	        const Eigen::Index dimension, const Eigen::Index vectorCount)
	{
		Detail::DavidsonInitialGuess::ValidateDimensions(dimension, vectorCount);
		return Eigen::MatrixX<Scalar>::Identity(dimension, vectorCount);
	}


	template <typename Scalar, typename Derived>
	[[nodiscard]] Eigen::MatrixX<Scalar> CoordinateDavidsonInitialBasis(
	        const Eigen::MatrixBase<Derived>& diagonal, const Eigen::Index vectorCount)
	{
		Detail::DavidsonInitialGuess::ValidateDimensions(diagonal.size(), vectorCount);
		if (!diagonal.allFinite())
		{
			throw InvalidArgumentException("A Davidson initial-basis diagonal must contain only finite values");
		}
		std::vector<Eigen::Index> indices(static_cast<std::size_t>(diagonal.size()));
		std::iota(indices.begin(), indices.end(), Eigen::Index{0});
		std::stable_sort(indices.begin(), indices.end(), [&diagonal](const Eigen::Index lhs, const Eigen::Index rhs)
		                 { return diagonal[lhs] < diagonal[rhs]; });
		Eigen::MatrixX<Scalar> basis = Eigen::MatrixX<Scalar>::Zero(diagonal.size(), vectorCount);
		for (Eigen::Index columnIndex = 0; columnIndex < vectorCount; columnIndex++)
		{
			basis(indices[static_cast<std::size_t>(columnIndex)], columnIndex) = Scalar{1};
		}
		return basis;
	}


	template <typename Scalar>
	[[nodiscard]] Eigen::MatrixX<Scalar> SeededRandomDavidsonInitialBasis(
	        const Eigen::Index dimension, const Eigen::Index vectorCount, const std::uint64_t seed)
	{
		Detail::DavidsonInitialGuess::ValidateDimensions(dimension, vectorCount);
		std::mt19937_64 generator(seed);
		std::normal_distribution<double> distribution;
		Eigen::MatrixX<Scalar> basis(dimension, vectorCount);
		for (Eigen::Index columnIndex = 0; columnIndex < vectorCount; columnIndex++)
		{
			for (Eigen::Index rowIndex = 0; rowIndex < dimension; rowIndex++)
			{
				basis(rowIndex, columnIndex) =
				        Detail::DavidsonInitialGuess::RandomScalar<Scalar>(generator, distribution);
			}
		}
		Eigen::HouseholderQR<Eigen::MatrixX<Scalar>> qr(basis);
		return qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(dimension, vectorCount);
	}


	template <typename Derived>
	[[nodiscard]] Eigen::MatrixX<typename Derived::Scalar> AugmentDavidsonInitialBasis(
	        const Eigen::MatrixBase<Derived>& suppliedBasis,
	        const Eigen::Index targetVectorCount,
	        const typename Eigen::NumTraits<typename Derived::Scalar>::Real linearDependenceTolerance)
	{
		using Scalar = Derived::Scalar;
		using RealScalar = Eigen::NumTraits<Scalar>::Real;
		Detail::DavidsonInitialGuess::ValidateDimensions(suppliedBasis.rows(), targetVectorCount);
		if (suppliedBasis.cols() <= 0 || suppliedBasis.cols() > suppliedBasis.rows()
		    || targetVectorCount < suppliedBasis.cols() || !suppliedBasis.allFinite())
		{
			throw InvalidArgumentException("The supplied Davidson basis has invalid dimensions or values");
		}
		if (!std::isfinite(linearDependenceTolerance) || linearDependenceTolerance <= RealScalar{0})
		{
			throw InvalidArgumentException("The Davidson linear-dependence tolerance must be finite and positive");
		}

		Eigen::MatrixX<Scalar> result = Eigen::MatrixX<Scalar>::Zero(suppliedBasis.rows(), targetVectorCount);
		Eigen::Index retainedCount = 0;
		auto appendIfIndependent = [&](Eigen::VectorX<Scalar> candidate)
		{
			if (retainedCount != 0)
			{
				candidate -= result.leftCols(retainedCount) * (result.leftCols(retainedCount).adjoint() * candidate);
				candidate -= result.leftCols(retainedCount) * (result.leftCols(retainedCount).adjoint() * candidate);
			}
			const RealScalar norm = candidate.norm();
			if (norm > linearDependenceTolerance)
			{
				result.col(retainedCount++) = candidate / norm;
			}
		};
		for (Eigen::Index columnIndex = 0;
		     columnIndex < suppliedBasis.cols() && retainedCount < targetVectorCount;
		     columnIndex++)
		{
			appendIfIndependent(suppliedBasis.col(columnIndex));
		}
		for (Eigen::Index coordinateIndex = 0;
		     coordinateIndex < suppliedBasis.rows() && retainedCount < targetVectorCount;
		     coordinateIndex++)
		{
			appendIfIndependent(Eigen::VectorX<Scalar>::Unit(suppliedBasis.rows(), coordinateIndex));
		}
		if (retainedCount != targetVectorCount)
		{
			throw InvariantViolationException("Coordinate augmentation failed to span the requested Davidson basis");
		}
		return result;
	}
}
