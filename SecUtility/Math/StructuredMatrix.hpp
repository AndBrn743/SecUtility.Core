// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Andy Brown

#pragma once

#include <Eigen/Dense>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Misc/Random.hpp>
#include <utility>


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

	/// <summary>
	/// First cols columns of a Householder-type unitary whose first column is parallel to the given
	/// vector.
	/// </summary>
	/// <param name="vector">Desired first column; normalized in place.</param>
	/// <param name="cols">Columns to return, in [1, vector.size()].</param>
	/// <returns>rows x cols matrix with orthonormal columns; first column parallel to vector.</returns>
	/// <remarks>
	/// Constructed via the Householder formula I - (2/|v|^2) v v*. For real Scalar the result is
	/// Hermitian (a true reflection). For complex Scalar the construction multiplies by a phase
	/// alpha = vector[0]/|vector[0]| to pin the first column, so the result is unitary but not
	/// Hermitian unless Im(vector[0]) = 0. For guaranteed-Hermitian complex output use
	/// FirstNColumnsOfRandomUnitaryHermitianWithGivenFirstColumn. Not Haar-random.
	/// </remarks>
	template <typename Scalar>
	Eigen::MatrixX<Scalar> FirstNColumnsOfHouseholderUnitaryWithGivenFirstColumn(Eigen::VectorX<Scalar> vector,
	                                                                             const Eigen::Index cols)
	{
		eigen_assert(cols > 0);
		eigen_assert(cols <= vector.size());
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

	/// <summary>A random square Householder-type unitary of the given dimension.</summary>
	/// <remarks>
	/// Hermitian for real Scalar; for complex Scalar the result is unitary but not Hermitian in
	/// general (random complex phase at vector[0]). For guaranteed-Hermitian complex output use
	/// RandomUnitaryHermitianMatrix. Not Haar-random.
	/// </remarks>
	template <typename Scalar>
	Eigen::MatrixX<Scalar> RandomHouseholderUnitary(const Eigen::Index dimension)
	{
		return FirstNColumnsOfHouseholderUnitaryWithGivenFirstColumn<Scalar>(Eigen::VectorX<Scalar>::Random(dimension),
		                                                                     dimension);
	}

	/// <summary>
	/// First cols columns of a Hermitian unitary (Householder reflector) whose first column is
	/// parallel to the given vector.
	/// </summary>
	/// <param name="vector">Desired first column; normalized in place.</param>
	/// <param name="cols">Columns to return, in [1, vector.size()].</param>
	/// <returns>rows x cols matrix with orthonormal columns; first column parallel to vector.</returns>
	/// <remarks>
	/// The result is Hermitian (a single Householder reflector) - a measure-zero subset of U(n).
	/// Not Haar-random by construction.
	/// </remarks>
	template <typename Scalar>
	Eigen::MatrixX<Scalar> FirstNColumnsOfRandomUnitaryHermitianWithGivenFirstColumn(Eigen::VectorX<Scalar> vector,
	                                                                                 const Eigen::Index cols)
	{
		eigen_assert(cols > 0);
		eigen_assert(cols <= vector.size());
		const auto rows = vector.size();
		eigen_assert(rows > 0);

		vector.normalize();

		if constexpr (!std::is_same_v<Scalar, typename Eigen::NumTraits<Scalar>::Real>)
		{
			eigen_assert(Math::Abs(Math::Im(vector[0])) < 1e-15);
			vector[0] = Math::Re(vector[0]);
		}

		return FirstNColumnsOfHouseholderUnitaryWithGivenFirstColumn(std::move(vector), cols);
	}

	/// <summary>Square Hermitian unitary with the given first column.</summary>
	/// <remarks>Hermitian (measure-zero subset of U(n)); not Haar-random. See
	/// FirstNColumnsOfRandomUnitaryHermitianWithGivenFirstColumn.</remarks>
	template <typename Scalar>
	Eigen::MatrixX<Scalar> RandomUnitaryHermitianWithGivenFirstColumn(Eigen::VectorX<Scalar> vector)
	{
		return FirstNColumnsOfRandomUnitaryHermitianWithGivenFirstColumn(vector, vector.size());
	}

	/// <summary>A random square Hermitian unitary of the given dimension.</summary>
	/// <remarks>Hermitian (measure-zero subset of U(n)); not Haar-random.</remarks>
	template <typename Scalar>
	Eigen::MatrixX<Scalar> RandomUnitaryHermitianMatrix(const Eigen::Index dimension)
	{
		Eigen::VectorX<Scalar> v = Eigen::VectorX<Scalar>::Random(dimension);
		v[0] = Math::Re(v[0]);
		return FirstNColumnsOfHouseholderUnitaryWithGivenFirstColumn(std::move(v), dimension);
	}

	/// <summary>
	/// First cols columns of a random unitary whose first column is parallel to the given vector.
	/// </summary>
	/// <param name="vector">Desired first column; normalized in place.</param>
	/// <param name="cols">Columns to return, in [1, vector.size()].</param>
	/// <returns>rows x cols matrix with orthonormal columns; first column parallel to vector.</returns>
	/// <remarks>
	/// NOT Haar-random. Built as H * diag(1, U) where H is the Householder reflector mapping e_1
	/// to vector/|vector| and U is a random Hermitian unitary. For cols &lt; rows, columns 2..cols
	/// are confined to the specific (cols-1)-dim subspace H * span{e_2, ..., e_cols} of v-perp,
	/// rather than being drawn from a random subspace. Fine for fixtures needing "some" random
	/// unitary; not appropriate for statistical tests that depend on the Haar measure.
	/// </remarks>
	template <typename Scalar>
	Eigen::MatrixX<Scalar> FirstNColumnsOfRandomUnitaryWithGivenFirstColumn(Eigen::VectorX<Scalar> vector,
	                                                                        const Eigen::Index cols)
	{
		eigen_assert(cols > 0);
		eigen_assert(cols <= vector.size());

		if (vector.size() == 1)
		{
			return Eigen::Matrix<Scalar, 1, 1>::Ones();
		}

		Eigen::MatrixX<Scalar> uh = FirstNColumnsOfHouseholderUnitaryWithGivenFirstColumn<Scalar>(vector, cols);

		if (cols <= 2)
		{
			return uh;
		}

		Eigen::MatrixX<Scalar> deHermitianlizer(cols, cols);
		deHermitianlizer.row(0).setZero();
		deHermitianlizer.col(0).setZero();
		deHermitianlizer(0, 0) = Scalar{1};
		deHermitianlizer.bottomRightCorner(cols - 1, cols - 1) = RandomHouseholderUnitary<Scalar>(cols - 1);

		return uh * deHermitianlizer;
	}

	/// <summary>Square random unitary with the given first column.</summary>
	/// <remarks>Not Haar-random; see FirstNColumnsOfRandomUnitaryWithGivenFirstColumn.</remarks>
	template <typename Scalar>
	Eigen::MatrixX<Scalar> RandomUnitaryWithGivenFirstColumn(Eigen::VectorX<Scalar> vector)
	{
		return FirstNColumnsOfRandomUnitaryWithGivenFirstColumn(vector, vector.size());
	}

	/// <summary>A random square unitary of the given dimension.</summary>
	/// <remarks>
	/// NOT Haar-random. Built as the product of two independent random Hermitian unitaries.
	/// For real Scalar the output is further restricted to SO(n) (determinant +1).
	/// </remarks>
	template <typename Scalar>
	Eigen::MatrixX<Scalar> RandomUnitaryMatrix(const Eigen::Index dimension)
	{
		return FirstNColumnsOfHouseholderUnitaryWithGivenFirstColumn<Scalar>(Eigen::VectorX<Scalar>::Random(dimension),
		                                                                     dimension)
		       * FirstNColumnsOfHouseholderUnitaryWithGivenFirstColumn<Scalar>(
		               Eigen::VectorX<Scalar>::Random(dimension), dimension);
	}

	/// <summary>First cols columns of a rows x rows random unitary.</summary>
	/// <remarks>Not Haar-random; see FirstNColumnsOfRandomUnitaryWithGivenFirstColumn.</remarks>
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
