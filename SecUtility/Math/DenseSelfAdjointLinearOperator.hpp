// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>

#include <Eigen/Core>

#include <utility>


namespace SecUtility::Math
{
	template <typename T>
	class DenseSelfAdjointLinearOperator
	{
	public:
		using Scalar = T;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

		explicit DenseSelfAdjointLinearOperator(Eigen::MatrixX<Scalar> matrix) : m_Matrix(std::move(matrix))
		{
			if (m_Matrix.rows() <= 0 || m_Matrix.rows() != m_Matrix.cols())
			{
				throw InvalidArgumentException("A dense self-adjoint operator must be nonempty and square");
			}
			if (!m_Matrix.allFinite() || !m_Matrix.isApprox(m_Matrix.adjoint()))
			{
				throw InvalidArgumentException("A dense self-adjoint operator must contain a finite self-adjoint matrix");
			}
		}

		[[nodiscard]] Eigen::Index rows() const noexcept { return m_Matrix.rows(); }
		[[nodiscard]] Eigen::Index cols() const noexcept { return m_Matrix.cols(); }
		[[nodiscard]] Eigen::VectorX<RealScalar> Diagonal() const { return m_Matrix.diagonal().real(); }
		[[nodiscard]] const Eigen::MatrixX<Scalar>& Matrix() const noexcept { return m_Matrix; }

		template <typename Derived>
		[[nodiscard]] auto ApplyOn(const Eigen::MatrixBase<Derived>& vectors) const
		{
			return (m_Matrix * vectors).eval();
		}

	private:
		Eigen::MatrixX<Scalar> m_Matrix;
	};


	template <typename Derived>
	DenseSelfAdjointLinearOperator(const Eigen::MatrixBase<Derived>&)
	        -> DenseSelfAdjointLinearOperator<typename Derived::Scalar>;
}
