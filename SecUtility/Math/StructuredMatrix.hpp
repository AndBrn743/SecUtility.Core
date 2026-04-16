// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Andy Brown

#pragma once

#include <Eigen/Dense>
#include <SecUtility/Misc/Random.hpp>


namespace SecUtility::Math
{
	template <typename Scalar>
	Eigen::MatrixX<Scalar> RandomSymmetricMatrix(const Eigen::Index dimension)
	{
		const Eigen::MatrixX<Scalar> m = Eigen::MatrixX<Scalar>::Random(dimension, dimension);
		return m + m.transpose();
	}

	template <typename Scalar>
	Eigen::MatrixX<Scalar> RandomHermitianMatrix(const Eigen::Index dimension)
	{
		const Eigen::MatrixX<Scalar> m = Eigen::MatrixX<Scalar>::Random(dimension, dimension);
		return m + m.adjoint();
	}

	template <typename Scalar>
	Eigen::MatrixX<Scalar> RandomAntiSymmetricMatrix(const Eigen::Index dimension)
	{
		const Eigen::MatrixX<Scalar> m = Eigen::MatrixX<Scalar>::Random(dimension, dimension);
		return m - m.transpose();
	}

	template <typename Scalar>
	Eigen::MatrixX<Scalar> RandomAntiHermitianMatrix(const Eigen::Index dimension)
	{
		const Eigen::MatrixX<Scalar> m = Eigen::MatrixX<Scalar>::Random(dimension, dimension);
		return m - m.adjoint();
	}

	template <typename Scalar>
	Eigen::MatrixX<Scalar> FirstNColumnsOfRandomUnitaryWithGivenFirstColumn(Eigen::VectorX<Scalar> vector,
	                                                                        const Eigen::Index cols)
	{
		eigen_assert(cols > 0);
		const auto rows = vector.size();
		eigen_assert(rows > 0);

		vector.normalize();

		const auto alpha =
		        std::is_same_v<Scalar, typename Eigen::NumTraits<Scalar>::Real> ? 1 : vector[0] / std::abs(vector[0]);
		vector[0] -= alpha;
		const auto squaredNorm = vector.squaredNorm();

		if (squaredNorm < static_cast<typename Eigen::NumTraits<Scalar>::Real>(1e-14))
		{
			return Eigen::MatrixX<Scalar>::Identity(rows, cols);
		}

		if constexpr (std::is_same_v<Scalar, typename Eigen::NumTraits<Scalar>::Real>)
		{
			return Eigen::MatrixX<Scalar>::Identity(rows, cols)
			       - (static_cast<Scalar>(2) / squaredNorm) * (vector * vector.head(cols).adjoint());
		}
		else
		{
			return (Eigen::MatrixX<Scalar>::Identity(rows, cols)
			        - (static_cast<Scalar>(2) / squaredNorm) * (vector * vector.head(cols).adjoint()))
			       * alpha;
		}
	}

	template <typename Scalar>
	Eigen::MatrixX<Scalar> RandomUnitaryWithGivenFirstColumn(Eigen::VectorX<Scalar> vector)
	{
		return FirstNColumnsOfRandomUnitaryWithGivenFirstColumn(vector, vector.size());
	}

	template <typename Scalar>
	Eigen::MatrixX<Scalar> RandomUnitaryMatrix(const Eigen::Index dimension)
	{
		return RandomUnitaryWithGivenFirstColumn<Scalar>(Eigen::VectorX<Scalar>::Random(dimension));
	}

	template <typename Scalar>
	Eigen::MatrixX<Scalar> FirstNColumnsOfRandomUnitaryMatrix(const Eigen::Index rows, const Eigen::Index cols)
	{
		return FirstNColumnsOfRandomUnitaryWithGivenFirstColumn<Scalar>(Eigen::VectorX<Scalar>::Random(rows), cols);
	}

	template <typename Scalar>
	Eigen::MatrixX<Scalar> RandomPositiveDefiniteHermitianMatrix(const Eigen::Index dimension)
	{
		Eigen::VectorX<Scalar> eigenvalues =
		        Eigen::VectorX<Scalar>::Random(dimension).cwiseAbs2() + 1e-2 * Eigen::VectorX<Scalar>::Ones(dimension);
		const auto eigenvectors = RandomUnitaryMatrix<Scalar>(dimension);
		return eigenvectors * eigenvalues.asDiagonal() * eigenvectors.adjoint();
	}

	template <typename Scalar>
	Eigen::MatrixX<Scalar> RandomSparseMatrixInDenseForm(const Eigen::Index rows,
	                                                     const Eigen::Index cols,
	                                                     const double density = 0.1,
	                                                     const typename Eigen::NumTraits<Scalar>::Real max = 1,
	                                                     const typename Eigen::NumTraits<Scalar>::Real min = -1)
	{
		Eigen::MatrixX<Scalar> m = Eigen::MatrixX<Scalar>::Zero(rows, cols);

		for (auto ptr = m.data(); ptr != m.data() + m.size(); ptr++)
		{
			if (Random::NextDouble() < density)
			{
				using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

				if constexpr (std::is_same_v<Scalar, RealScalar>)
				{
					*ptr = static_cast<Scalar>(Random::NextDouble(min, max));
				}
				else
				{
					*ptr = Scalar{Random::NextDouble(min, max), Random::NextDouble(min, max)};
				}
			}
		}

		return m;
	}
}  // namespace SecUtility::Math
