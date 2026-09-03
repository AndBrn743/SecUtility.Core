// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/BlockExpressionBase.hpp>
#include <SecUtility/Hoppy/ForwardDeclarations.hpp>

#include <Eigen/Core>
#include <Eigen/LU>

namespace Hoppy
{
	template <typename Derived>
	class BlockDiagonalMatrixExpr : public BlockExpressionBase<Derived>
	{
	public:
		using Base = BlockExpressionBase<Derived>;
		using Scalar = typename Base::Scalar;
		using Base::derived;

		auto operator+() const&;
		auto operator-() const&;
		auto transpose() const&;
		auto conjugate() const&;
		auto adjoint() const&;
		auto real() const&;
		auto imag() const&;
		template <typename NewScalar> auto cast() const&;
		template <typename NewScalar> auto cast() const&& = delete;
		auto operator+() const&& = delete;
		auto operator-() const&& = delete;
		auto transpose() const&& = delete;
		auto conjugate() const&& = delete;
		auto adjoint() const&& = delete;
		auto real() const&& = delete;
		auto imag() const&& = delete;

		Scalar trace() const { Scalar result{}; for (Eigen::Index i = 0; i < this->blockCount(); ++i) result += derived()[i].trace(); return result; }
		Scalar determinant() const { Scalar result{1}; for (Eigen::Index i = 0; i < this->blockCount(); ++i) result *= derived()[i].determinant(); return result; }
		template <typename S = Scalar, typename = std::enable_if_t<!Eigen::NumTraits<S>::IsComplex>>
		Scalar maxCoeff() const { Scalar result = Base::template maxCoeff<S>(); return this->blockCount() > 1 ? (std::max)(result, Scalar{}) : result; }
		template <typename S = Scalar, typename = std::enable_if_t<!Eigen::NumTraits<S>::IsComplex>>
		Scalar minCoeff() const { Scalar result = Base::template minCoeff<S>(); return this->blockCount() > 1 ? (std::min)(result, Scalar{}) : result; }

		BlockDiagonalMatrix<Scalar> eval() const { return BlockDiagonalMatrix<Scalar>(derived()); }

		Eigen::MatrixX<Scalar> toDense() const
		{
			Eigen::MatrixX<Scalar> result(this->rows(), this->cols());
			evalTo(result);
			return result;
		}

		template <typename Destination>
		void evalTo(Eigen::MatrixBase<Destination>& destination) const
		{
			Detail::assertNoOverlap(derived(), destination.derived());
			destination.derived().resize(this->rows(), this->cols());
			destination.derived().setZero();
			for (Eigen::Index index = 0; index < this->blockCount(); ++index)
			{
				const auto offset = this->blockOffset(index);
				const auto dimension = this->dimensionOfBlock(index);
				destination.derived().block(offset, offset, dimension, dimension) = derived()[index];
			}
		}
	};
}  // namespace Hoppy
