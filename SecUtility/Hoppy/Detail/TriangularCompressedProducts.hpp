// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/Detail/TriangularCompressedExpressions.hpp>

namespace Hoppy::Detail
{
	template <typename Left, typename Right>
	using product_scalar_t = typename Eigen::ScalarBinaryOpTraits<
	        typename std::remove_cv_t<std::remove_reference_t<Left>>::Scalar,
	        typename std::remove_cv_t<std::remove_reference_t<Right>>::Scalar,
	        Eigen::internal::scalar_product_op<
	                typename std::remove_cv_t<std::remove_reference_t<Left>>::Scalar,
		                typename std::remove_cv_t<std::remove_reference_t<Right>>::Scalar>>::ReturnType;
	template <typename Left, typename Right>
	class DenseProduct;
}

namespace Eigen::internal
{
	template <typename Left, typename Right>
	struct traits<Hoppy::Detail::DenseProduct<Left, Right>>
	{
		using Lhs = std::remove_cv_t<std::remove_reference_t<Left>>;
		using Rhs = std::remove_cv_t<std::remove_reference_t<Right>>;
		using Scalar = Hoppy::Detail::product_scalar_t<Left, Right>;
		using ReturnType = Eigen::Matrix<Scalar, Lhs::RowsAtCompileTime, Rhs::ColsAtCompileTime>;
		using StorageKind = Dense;
		using XprKind = MatrixXpr;
		using StorageIndex = Eigen::Index;
		static constexpr int Flags = EvalBeforeNestingBit;
		static constexpr int RowsAtCompileTime = Lhs::RowsAtCompileTime;
		static constexpr int ColsAtCompileTime = Rhs::ColsAtCompileTime;
		static constexpr int MaxRowsAtCompileTime = Lhs::RowsAtCompileTime;
		static constexpr int MaxColsAtCompileTime = Rhs::ColsAtCompileTime;
	};
}

namespace Hoppy::Detail
{

	template <typename Left, typename Right>
	class TriangularCompressedProduct
	    : public Hoppy::TriangularCompressedMatrixExpr<TriangularCompressedProduct<Left, Right>>
	{
		using Lhs = std::remove_cv_t<std::remove_reference_t<Left>>;
	public:
		using Scalar = product_scalar_t<Left, Right>;
		using StructureTag = typename Lhs::StructureTag;
		static constexpr int RowsAtCompileTime = Lhs::RowsAtCompileTime;
		static constexpr int ColsAtCompileTime = Lhs::ColsAtCompileTime;
		static constexpr TrianglePacking PackingValue = Lhs::PackingValue;
		static constexpr bool IsTriangularCompressed = true;
		static constexpr bool IsWritable = false;
		using PlainObject = TriangularCompressedMatrix<Scalar, RowsAtCompileTime, PackingValue, 0, StructureTag>;
		TriangularCompressedProduct(Left left, Right right)
			: m_Left(std::forward<Left>(left)), m_Right(std::forward<Right>(right))
		{ eigen_assert(m_Left.cols() == m_Right.rows()); }
		Eigen::Index dimension() const { return m_Left.dimension(); }
		Eigen::Index rows() const { return dimension(); }
		Eigen::Index cols() const { return dimension(); }
		Eigen::Index size() const { return dimension() * dimension(); }
		Scalar coeff(Eigen::Index row, Eigen::Index column) const
		{
			Scalar result(0);
			for (Eigen::Index index = 0; index < dimension(); ++index)
				result += m_Left.coeff(row, index) * m_Right.coeff(index, column);
			return result;
		}
		auto toDense() const { return evalDense(*this); }
		PlainObject eval() const { return PlainObject(*this); }
	private: Left m_Left; Right m_Right;
	};

	template <typename Left, typename Right>
	class DenseProduct : public Eigen::MatrixBase<DenseProduct<Left, Right>>
	{
	public:
		using This = DenseProduct;
		using Base = Eigen::MatrixBase<This>;
		EIGEN_DENSE_PUBLIC_INTERFACE(This)
		using ReturnType = typename Eigen::internal::traits<DenseProduct>::ReturnType;
		static constexpr bool IsTriangularCompressed = false;
		static constexpr bool IsDensePromoted = true;
		DenseProduct(Left left, Right right)
			: m_Left(std::forward<Left>(left)), m_Right(std::forward<Right>(right))
		{ eigen_assert(m_Left.cols() == m_Right.rows()); }
		Eigen::Index rows() const { return m_Left.rows(); }
		Eigen::Index cols() const { return m_Right.cols(); }
		template <typename Destination>
		void evalTo(Destination& destination) const
		{
			decltype(auto) left = coefficientOperand(m_Left);
			decltype(auto) right = coefficientOperand(m_Right);
			ReturnType evaluated(rows(), cols());
			for (Eigen::Index row = 0; row < rows(); ++row)
				for (Eigen::Index column = 0; column < cols(); ++column)
				{
					Scalar result(0);
					for (Eigen::Index index = 0; index < m_Left.cols(); ++index)
						result += left.coeff(row, index) * right.coeff(index, column);
					evaluated.coeffRef(row, column) = result;
				}
			destination = evaluated;
		}
		ReturnType toDense() const
		{
			ReturnType result(rows(), cols());
			evalTo(result);
			return result;
		}
	private: Left m_Left; Right m_Right;
	};

	template <typename Left, typename Right>
	inline constexpr bool structured_product_v =
	        (std::is_same_v<typename std::remove_cv_t<std::remove_reference_t<Left>>::StructureTag, UpperTriangularTag>
	         && std::is_same_v<typename std::remove_cv_t<std::remove_reference_t<Right>>::StructureTag, UpperTriangularTag>)
	        || (std::is_same_v<typename std::remove_cv_t<std::remove_reference_t<Left>>::StructureTag, LowerTriangularTag>
	            && std::is_same_v<typename std::remove_cv_t<std::remove_reference_t<Right>>::StructureTag, LowerTriangularTag>);
}

namespace Hoppy
{
	template <typename Derived> template <typename Other, typename, typename>
	auto TriangularCompressedMatrixExpr<Derived>::operator*(Other&& other) const&
	{
		using StoredRight = Detail::nested_operand_t<Other&&>;
		if constexpr (Detail::structured_product_v<Derived, StoredRight>)
			return Detail::TriangularCompressedProduct<const Derived&, StoredRight>(derived(), std::forward<Other>(other));
		else return Detail::DenseProduct<const Derived&, StoredRight>(derived(), std::forward<Other>(other));
	}
	template <typename Derived> template <typename Other, typename, typename>
	auto TriangularCompressedMatrixExpr<Derived>::operator*(Other&& other) &&
	{
		using StoredRight = Detail::nested_operand_t<Other&&>;
		if constexpr (Detail::structured_product_v<Derived, StoredRight>)
			return Detail::TriangularCompressedProduct<Derived, StoredRight>(std::move(derived()), std::forward<Other>(other));
		else return Detail::DenseProduct<Derived, StoredRight>(std::move(derived()), std::forward<Other>(other));
	}
	template <typename Derived> template <typename Dense, typename, typename, typename>
	auto TriangularCompressedMatrixExpr<Derived>::operator*(Dense&& other) const&
	{
		using StoredRight = Detail::nested_operand_t<Dense&&>;
		return Detail::DenseProduct<const Derived&, StoredRight>(derived(), std::forward<Dense>(other));
	}
	template <typename Derived> template <typename Dense, typename, typename, typename>
	auto TriangularCompressedMatrixExpr<Derived>::operator*(Dense&& other) &&
	{
		using StoredRight = Detail::nested_operand_t<Dense&&>;
		return Detail::DenseProduct<Derived, StoredRight>(std::move(derived()), std::forward<Dense>(other));
	}

	template <typename Dense, typename Derived,
	          typename = std::enable_if_t<Detail::is_dense_matrix_operand_v<Dense>>, typename = void>
	auto operator*(Dense&& dense, const TriangularCompressedMatrixExpr<Derived>& expression)
	{
		using StoredLeft = Detail::nested_operand_t<Dense&&>;
		return Detail::DenseProduct<StoredLeft, const Derived&>(std::forward<Dense>(dense), expression.derived());
	}
}

namespace Eigen::internal
{
	template <typename Left, typename Right>
	struct evaluator<Hoppy::Detail::DenseProduct<Left, Right>>
	    : triangular_dense_evaluator<Hoppy::Detail::DenseProduct<Left, Right>>
	{
		using Expression = Hoppy::Detail::DenseProduct<Left, Right>;
		using Base = triangular_dense_evaluator<Expression>;
		explicit evaluator(const Expression& expression) : Base(expression) {}
	};
	template <typename Left, typename Right>
	struct traits<Hoppy::Detail::TriangularCompressedProduct<Left, Right>>
	    : triangular_node_traits<Hoppy::Detail::TriangularCompressedProduct<Left, Right>> {};
	template <typename Left, typename Right>
	struct evaluator<Hoppy::Detail::TriangularCompressedProduct<Left, Right>>
	    : triangular_compressed_evaluator<Hoppy::Detail::TriangularCompressedProduct<Left, Right>>
	{
		using Expression = Hoppy::Detail::TriangularCompressedProduct<Left, Right>;
		using Base = triangular_compressed_evaluator<Expression>;
		explicit evaluator(const Expression& expression) : Base(expression) {}
	};
}
