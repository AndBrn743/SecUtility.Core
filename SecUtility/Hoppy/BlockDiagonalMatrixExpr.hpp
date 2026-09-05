// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/BlockExpressionBase.hpp>
#include <SecUtility/Hoppy/Detail/Traits.hpp>
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
		template <typename OtherDerived,
		          typename = typename Eigen::ScalarBinaryOpTraits<
		                  Scalar, typename Eigen::internal::traits<OtherDerived>::Scalar,
		                  Eigen::internal::scalar_sum_op<Scalar,
		                                                 typename Eigen::internal::traits<OtherDerived>::Scalar>>::ReturnType>
		auto operator+(const BlockDiagonalMatrixExpr<OtherDerived>& other) const&;
		template <typename OtherDerived,
		          typename = typename Eigen::ScalarBinaryOpTraits<
		                  Scalar, typename Eigen::internal::traits<OtherDerived>::Scalar,
		                  Eigen::internal::scalar_difference_op<
		                          Scalar, typename Eigen::internal::traits<OtherDerived>::Scalar>>::ReturnType>
		auto operator-(const BlockDiagonalMatrixExpr<OtherDerived>& other) const&;
		template <typename TOtherScalar,
		          typename = typename Eigen::ScalarBinaryOpTraits<
		                  Scalar, TOtherScalar,
		                  Eigen::internal::scalar_product_op<Scalar, TOtherScalar>>::ReturnType>
		auto operator*(const TOtherScalar& scalar) const&;
		template <typename TOtherScalar,
		          typename = typename Eigen::ScalarBinaryOpTraits<
		                  Scalar, TOtherScalar,
		                  Eigen::internal::scalar_quotient_op<Scalar, TOtherScalar>>::ReturnType>
		auto operator/(const TOtherScalar& scalar) const&;
		template <typename OtherDerived,
		          typename = typename Eigen::ScalarBinaryOpTraits<
		                  Scalar, typename Eigen::internal::traits<OtherDerived>::Scalar,
		                  Eigen::internal::scalar_product_op<
		                          Scalar, typename Eigen::internal::traits<OtherDerived>::Scalar>>::ReturnType>
		auto operator*(const BlockDiagonalMatrixExpr<OtherDerived>& other) const&;
		template <typename OtherDerived> auto operator+(const BlockDiagonalMatrixExpr<OtherDerived>& other) const&& = delete;
		template <typename OtherDerived> auto operator-(const BlockDiagonalMatrixExpr<OtherDerived>& other) const&& = delete;
		template <typename TOtherScalar,
		          typename = typename Eigen::ScalarBinaryOpTraits<
		                  Scalar, TOtherScalar,
		                  Eigen::internal::scalar_product_op<Scalar, TOtherScalar>>::ReturnType>
		auto operator*(const TOtherScalar& scalar) const&& = delete;
		template <typename TOtherScalar> auto operator/(const TOtherScalar& scalar) const&& = delete;
		template <typename OtherDerived> auto operator*(const BlockDiagonalMatrixExpr<OtherDerived>& other) const&& = delete;
		template <typename S = Scalar,
		          typename = std::enable_if_t<
		                  std::is_floating_point_v<typename Eigen::NumTraits<S>::Real>>>
		auto inverse() const&;
		template <typename S = Scalar,
		          typename = std::enable_if_t<
		                  std::is_floating_point_v<typename Eigen::NumTraits<S>::Real>>>
		auto inverse() const&& = delete;
		template <typename TransformDerived,
		          typename TransformScalar = typename Eigen::internal::traits<TransformDerived>::Scalar,
		          typename = typename Detail::congruence_result_scalar<Scalar,
		                                                               TransformScalar>::type>
		auto transformedBy(const BlockDiagonalMatrixExpr<TransformDerived>& transform) const&;
		template <typename TransformDerived,
		          typename TransformScalar = typename Eigen::internal::traits<TransformDerived>::Scalar,
		          typename = typename Detail::congruence_result_scalar<Scalar,
		                                                               TransformScalar>::type>
		auto backTransformedBy(const BlockDiagonalMatrixExpr<TransformDerived>& transform) const&;
		template <typename TransformDerived>
		auto transformedBy(const BlockDiagonalMatrixExpr<TransformDerived>& transform) const&& = delete;
		template <typename TransformDerived>
		auto backTransformedBy(const BlockDiagonalMatrixExpr<TransformDerived>& transform) const&& = delete;
		auto diagonal() const&;
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
