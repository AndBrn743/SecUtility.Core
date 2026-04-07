// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 Andy Brown

#pragma once

#include <Eigen/Dense>
#include <SecUtility/Diagnostic/Exception.hpp>
#include <cassert>


namespace SecUtility::Math
{
	template <typename Matrix>
	void OrthogonalizeInplaceWithClassicalGramSchmidt(Matrix&& ref_matrix, const Eigen::Index startColumn = 0)
	{
		static_assert(std::is_base_of_v<Eigen::MatrixBase<std::decay_t<Matrix>>, std::decay_t<Matrix>>);
		static_assert(!std::is_const_v<std::decay_t<Matrix>>);
		static_assert(std::is_lvalue_reference_v<Matrix>
		              || Eigen::internal::traits<std::decay_t<Matrix>>::Flags & Eigen::LvalueBit);
		assert(startColumn >= 0);

		for (Eigen::Index i = startColumn; i < ref_matrix.cols(); i++)
		{
			// Replace inner loop over each previous vector in `ref_matrix` with fast matrix-vector multiplication
			if (i > 0)
			{
				ref_matrix.col(i) -= ref_matrix.leftCols(i) * (ref_matrix.leftCols(i).adjoint() * ref_matrix.col(i));
			}

			// Normalize vector if possible
			const auto norm = ref_matrix.col(i).norm();
			if (norm <= 1e-13)
			{
				throw RuntimeException("(Classical) Gram-Schmidt Orthogonalization", "Linear Dependence");
			}

			ref_matrix.col(i) /= norm;
		}
	}


	template <typename Matrix>
	void OrthogonalizeInplaceWithModifiedGramSchmidt(Matrix&& ref_matrix, const Eigen::Index startColumn = 0)
	{
		static_assert(std::is_base_of_v<Eigen::MatrixBase<std::decay_t<Matrix>>, std::decay_t<Matrix>>);
		static_assert(std::is_lvalue_reference_v<Matrix>
		              || Eigen::internal::traits<std::decay_t<Matrix>>::Flags & Eigen::LvalueBit);
		assert(startColumn >= 0);

		for (Eigen::Index i = startColumn; i < ref_matrix.cols(); i++)
		{
			// Replace inner loop over each previous vector in `result` with fast matrix-vector multiplication
			for (Eigen::Index j = 0; j < i; j++)
			{
				ref_matrix.col(i) -= ref_matrix.col(j).dot(ref_matrix.col(i)) * ref_matrix.col(j);
			}

			// Normalize vector if possible
			const auto norm = ref_matrix.col(i).norm();
			if (norm <= 1e-13)
			{
				throw RuntimeException("Modified Gram-Schmidt Orthogonalization", "Linear Dependence");
			}

			ref_matrix.col(i) /= norm;
		}
	}


	template <typename Scalar, auto... Args>
	void OrthogonalizeAndRemoveLinearDependenceInplaceWithModifiedGramSchmidt(
	        Eigen::Matrix<Scalar, Args...>& ref_matrix, const Eigen::Index startColumn = 0)
	{
		assert(startColumn >= 0);
		Eigen::Index linearIndependentColumnCount = ref_matrix.cols();

		for (Eigen::Index i = startColumn; i < linearIndependentColumnCount; /* NO CODE */)
		{
			// Replace inner loop over each previous vector in `result` with fast matrix-vector multiplication
			for (Eigen::Index j = 0; j < i; j++)
			{
				ref_matrix.col(i) -= ref_matrix.col(j).dot(ref_matrix.col(i)) * ref_matrix.col(j);
			}

			// Normalize vector if possible
			const auto norm = ref_matrix.col(i).norm();
			if (norm <= 1e-13)
			{
				linearIndependentColumnCount--;
				ref_matrix.col(i) = ref_matrix.col(linearIndependentColumnCount);
			}
			else
			{
				ref_matrix.col(i) /= norm;
				i++;
			}
		}

		ref_matrix.conservativeResize(Eigen::NoChange, linearIndependentColumnCount);
	}


	template <typename Matrix>
	void OrthogonalizeInplaceWithGramSchmidt(Matrix&& ref_matrix, const Eigen::Index startColumn = 0)
	{
		static_assert(std::is_base_of_v<Eigen::MatrixBase<std::decay_t<Matrix>>, std::decay_t<Matrix>>);
		OrthogonalizeInplaceWithClassicalGramSchmidt(ref_matrix, startColumn);
	}


	//----------------------------------------------------------------------------------------------------------------//


	template <typename Matrix>
	auto OrthogonalizedWithClassicalGramSchmidt(const Matrix& matrix, const Eigen::Index startColumn = 0)
	{
		static_assert(std::is_base_of_v<Eigen::MatrixBase<std::decay_t<Matrix>>, std::decay_t<Matrix>>);
		auto m = matrix.eval();
		OrthogonalizeInplaceWithClassicalGramSchmidt(m, startColumn);
		return m;
	}


	template <typename Matrix>
	auto OrthogonalizedWithModifiedGramSchmidt(const Matrix& matrix, const Eigen::Index startColumn = 0)
	{
		static_assert(std::is_base_of_v<Eigen::MatrixBase<std::decay_t<Matrix>>, std::decay_t<Matrix>>);
		auto m = matrix.eval();
		OrthogonalizeInplaceWithModifiedGramSchmidt(m, startColumn);
		return m;
	}


	template <typename Matrix>
	auto OrthogonalizedAndLinearDependenceRemovedWithModifiedGramSchmidt(const Matrix& matrix,
	                                                                     const Eigen::Index startColumn = 0)
	{
		static_assert(std::is_base_of_v<Eigen::MatrixBase<std::decay_t<Matrix>>, std::decay_t<Matrix>>);
		auto m = matrix.eval();
		OrthogonalizeAndRemoveLinearDependenceInplaceWithModifiedGramSchmidt(m, startColumn);
		return m;
	}


	//----------------------------------------------------------------------------------------------------------------//


	template <typename Matrix>
	Eigen::MatrixX<typename Eigen::internal::traits<Matrix>::Scalar> OrthogonalizedAndLinearDependenceRemovedWithQR(
	        const Eigen::MatrixBase<Matrix>& matrix)
	{
		using Scalar = typename Eigen::internal::traits<Matrix>::Scalar;

		const Eigen::ColPivHouseholderQR<Eigen::MatrixX<Scalar>> qr(matrix);
		return qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(matrix.rows(), qr.nonzeroPivots());
	}


	template <typename Matrix>
	Eigen::MatrixX<typename Eigen::internal::traits<Matrix>::Scalar> OrthogonalizedAndLinearDependenceRemovedWithQR(
	        const Eigen::MatrixBase<Matrix>& matrix, const Eigen::Index numberOfLeftColumnsToExclude)
	{
		using Scalar = typename Eigen::internal::traits<Matrix>::Scalar;

		assert(numberOfLeftColumnsToExclude >= 0);

		if (numberOfLeftColumnsToExclude == 0)
		{
			return OrthogonalizedAndLinearDependenceRemovedWithQR(matrix);
		}

		const auto fixedOrthonormal = matrix.leftCols(numberOfLeftColumnsToExclude);
		const auto toBeOrthonormalized = matrix.rightCols(matrix.cols() - numberOfLeftColumnsToExclude);

		// Step 1: project out the orthogonal complement
		Eigen::MatrixX<Scalar> projectedX =
		        toBeOrthonormalized - fixedOrthonormal * (fixedOrthonormal.transpose() * toBeOrthonormalized);

		// Step 2: QR on projected part
		const Eigen::ColPivHouseholderQR<Eigen::MatrixX<Scalar>> qr(projectedX);
		Eigen::MatrixX<Scalar> orthonormalized =
		        qr.householderQ() * Eigen::MatrixX<Scalar>::Identity(projectedX.rows(), qr.nonzeroPivots());

		// Step 3: concatenate
		Eigen::MatrixX<Scalar> result(fixedOrthonormal.rows(), fixedOrthonormal.cols() + orthonormalized.cols());
		result << fixedOrthonormal, orthonormalized;
		return result;
	}
}  // namespace SecUtility::Math
