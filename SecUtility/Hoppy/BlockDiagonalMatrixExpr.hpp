// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/BlockExpressionBase.hpp>
#include <SecUtility/Hoppy/ForwardDeclarations.hpp>

#include <Eigen/Core>

namespace Hoppy
{
	template <typename Derived>
	class BlockDiagonalMatrixExpr : public BlockExpressionBase<Derived>
	{
	public:
		using Base = BlockExpressionBase<Derived>;
		using Scalar = typename Base::Scalar;
		using Base::derived;

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
