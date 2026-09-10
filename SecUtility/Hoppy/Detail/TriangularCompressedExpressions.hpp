// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedTraits.hpp>
#include <SecUtility/Hoppy/TriangularCompressedMatrixExpr.hpp>

#include <Eigen/Core>

#include <type_traits>
#include <utility>

namespace Hoppy::Detail
{
	struct TransposeOperation {};
	struct ConjugateOperation {};
	struct AdjointOperation {};
	struct NegateOperation {};
	struct SumOperation {};
	struct DifferenceOperation {};
	struct ProductOperation {};
	struct QuotientOperation {};

	template <typename T>
	using nested_operand_t = std::conditional_t<std::is_lvalue_reference_v<T>,
	                                            const std::remove_reference_t<T>&,
	                                            std::remove_reference_t<T>>;

	template <typename Operation, typename Left, typename Right>
	using binary_scalar_t = typename Eigen::ScalarBinaryOpTraits<
	        Left, Right,
	        std::conditional_t<std::is_same_v<Operation, SumOperation>, Eigen::internal::scalar_sum_op<Left, Right>,
	        std::conditional_t<std::is_same_v<Operation, DifferenceOperation>, Eigen::internal::scalar_difference_op<Left, Right>,
	        std::conditional_t<std::is_same_v<Operation, ProductOperation>, Eigen::internal::scalar_product_op<Left, Right>,
	                           Eigen::internal::scalar_quotient_op<Left, Right>>>>>::ReturnType;

	template <typename Operation, typename Left, typename Right>
	auto applyBinary(const Left& left, const Right& right)
	{
		if constexpr (std::is_same_v<Operation, SumOperation>) return left + right;
		else if constexpr (std::is_same_v<Operation, DifferenceOperation>) return left - right;
		else if constexpr (std::is_same_v<Operation, ProductOperation>) return left * right;
		else return left / right;
	}
	template <typename Expression> auto evalDense(const Expression& expression);
	template <typename Operand>
	decltype(auto) coefficientOperand(const Operand& operand)
	{
		if constexpr (is_return_by_value_operand_v<Operand>
		              || is_dense_promoted_expression<
		                      std::remove_cv_t<std::remove_reference_t<Operand>>>::value)
			return operand.eval();
		else return (operand);
	}

	template <typename Source, typename Factor>
	inline constexpr bool preserves_scaled_structure_v =
	        (!std::is_same_v<typename Source::StructureTag, HermitianTag>
	         && !std::is_same_v<typename Source::StructureTag, AntiHermitianTag>)
	        || Eigen::NumTraits<std::remove_cv_t<std::remove_reference_t<Factor>>>::IsComplex == 0;

	template <typename Tag>
	using transposed_structure_tag_t = std::conditional_t<
	        std::is_same_v<Tag, UpperTriangularTag>, LowerTriangularTag,
	        std::conditional_t<std::is_same_v<Tag, LowerTriangularTag>, UpperTriangularTag, Tag>>;

	template <TrianglePacking Packing>
	inline constexpr TrianglePacking flipped_packing_v =
	        Packing == TrianglePacking::Lower ? TrianglePacking::Upper : TrianglePacking::Lower;

	template <typename Operand, typename Operation>
	class TriangularCompressedTransform
	    : public Hoppy::TriangularCompressedMatrixExpr<TriangularCompressedTransform<Operand, Operation>>
	{
		using Source = std::remove_cv_t<std::remove_reference_t<Operand>>;

	public:
		using Scalar = typename Source::Scalar;
		using StructureTag = std::conditional_t<
		        std::is_same_v<Operation, ConjugateOperation>, typename Source::StructureTag,
		        transposed_structure_tag_t<typename Source::StructureTag>>;
		static constexpr int RowsAtCompileTime = Source::RowsAtCompileTime;
		static constexpr int ColsAtCompileTime = Source::ColsAtCompileTime;
		static constexpr TrianglePacking PackingValue =
		        std::is_same_v<Operation, ConjugateOperation> ? Source::PackingValue
		                                                  : flipped_packing_v<Source::PackingValue>;
		static constexpr int Flags = 0;
		static constexpr bool IsTriangularCompressed = true;
		static constexpr bool IsWritable = false;
		using PlainObject = TriangularCompressedMatrix<Scalar, RowsAtCompileTime, PackingValue, 0,
		                                               StructureTag>;

		explicit TriangularCompressedTransform(Operand operand) : m_Operand(std::forward<Operand>(operand)) {}
		Eigen::Index dimension() const noexcept { return m_Operand.dimension(); }
		Eigen::Index rows() const noexcept { return dimension(); }
		Eigen::Index cols() const noexcept { return dimension(); }
		Eigen::Index size() const noexcept { return dimension() * dimension(); }
		Scalar coeff(Eigen::Index row, Eigen::Index column) const
		{
			if constexpr (std::is_same_v<Operation, TransposeOperation>)
				return m_Operand.coeff(column, row);
			else if constexpr (std::is_same_v<Operation, ConjugateOperation>)
				return Eigen::numext::conj(m_Operand.coeff(row, column));
			else return Eigen::numext::conj(m_Operand.coeff(column, row));
		}
		Scalar operator()(const Eigen::Index row, const Eigen::Index column) const { return coeff(row, column); }

		PlainObject eval() const { return PlainObject(*this); }
		auto toDense() const
		{
			Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime> result(rows(), cols());
			evalTo(result);
			return result;
		}
		template <typename Destination>
		void evalTo(Eigen::MatrixBase<Destination>& destination) const
		{
			Eigen::Matrix<Scalar, RowsAtCompileTime, ColsAtCompileTime> evaluated(rows(), cols());
			for (Eigen::Index row = 0; row < rows(); ++row)
				for (Eigen::Index column = 0; column < cols(); ++column)
					evaluated.coeffRef(row, column) = coeff(row, column);
			destination.derived() = evaluated;
		}
		template <typename Dense,
		          typename = std::enable_if_t<std::is_base_of_v<Eigen::MatrixBase<Dense>, Dense>>>
		/* IMPLICIT */ operator Dense() const
		{
			Dense result;
			evalTo(result);
			return result;
		}

	private:
		Operand m_Operand;
	};

	template <typename Operand, typename ResultScalar>
	class TriangularCompressedNegate
	    : public Hoppy::TriangularCompressedMatrixExpr<TriangularCompressedNegate<Operand, ResultScalar>>
	{
		using Source = std::remove_cv_t<std::remove_reference_t<Operand>>;
	public:
		using Scalar = ResultScalar;
		using StructureTag = typename Source::StructureTag;
		static constexpr int RowsAtCompileTime = Source::RowsAtCompileTime;
		static constexpr int ColsAtCompileTime = Source::ColsAtCompileTime;
		static constexpr TrianglePacking PackingValue = Source::PackingValue;
		static constexpr bool IsTriangularCompressed = true;
		static constexpr bool IsWritable = false;
		using PlainObject = TriangularCompressedMatrix<Scalar, RowsAtCompileTime, PackingValue, 0, StructureTag>;
		explicit TriangularCompressedNegate(Operand operand) : m_Operand(std::forward<Operand>(operand)) {}
		Eigen::Index dimension() const { return m_Operand.dimension(); }
		Eigen::Index rows() const { return dimension(); }
		Eigen::Index cols() const { return dimension(); }
		Eigen::Index size() const { return dimension() * dimension(); }
		Scalar coeff(Eigen::Index row, Eigen::Index column) const { return -m_Operand.coeff(row, column); }
		auto toDense() const { return evalDense(*this); }
		PlainObject eval() const { return PlainObject(*this); }
	private: Operand m_Operand;
	};

	template <typename Operand, typename NewScalar>
	class TriangularCompressedCast
	    : public Hoppy::TriangularCompressedMatrixExpr<TriangularCompressedCast<Operand, NewScalar>>
	{
		using Source = std::remove_cv_t<std::remove_reference_t<Operand>>;
	public:
		using Scalar = NewScalar;
		using StructureTag = normalized_structure_tag_t<Scalar, typename Source::StructureTag>;
		static constexpr int RowsAtCompileTime = Source::RowsAtCompileTime;
		static constexpr int ColsAtCompileTime = Source::ColsAtCompileTime;
		static constexpr TrianglePacking PackingValue = Source::PackingValue;
		static constexpr bool IsTriangularCompressed = true;
		static constexpr bool IsWritable = false;
		using PlainObject = TriangularCompressedMatrix<Scalar, RowsAtCompileTime, PackingValue, 0, StructureTag>;
		explicit TriangularCompressedCast(Operand operand) : m_Operand(std::forward<Operand>(operand)) {}
		Eigen::Index dimension() const { return m_Operand.dimension(); }
		Eigen::Index rows() const { return dimension(); }
		Eigen::Index cols() const { return dimension(); }
		Eigen::Index size() const { return dimension() * dimension(); }
		Scalar coeff(Eigen::Index row, Eigen::Index column) const
		{ return static_cast<Scalar>(m_Operand.coeff(row, column)); }
		auto toDense() const { return evalDense(*this); }
		PlainObject eval() const { return PlainObject(*this); }
	private: Operand m_Operand;
	};

	template <typename Left, typename Right, typename Operation>
	class TriangularCompressedBinary
	    : public Hoppy::TriangularCompressedMatrixExpr<TriangularCompressedBinary<Left, Right, Operation>>
	{
		using Lhs = std::remove_cv_t<std::remove_reference_t<Left>>;
		using Rhs = std::remove_cv_t<std::remove_reference_t<Right>>;
	public:
		using Scalar = binary_scalar_t<Operation, typename Lhs::Scalar, typename Rhs::Scalar>;
		using StructureTag = typename Lhs::StructureTag;
		static constexpr int RowsAtCompileTime = Lhs::RowsAtCompileTime;
		static constexpr int ColsAtCompileTime = Lhs::ColsAtCompileTime;
		static constexpr TrianglePacking PackingValue = Lhs::PackingValue;
		static constexpr bool IsTriangularCompressed = true;
		static constexpr bool IsWritable = false;
		using PlainObject = TriangularCompressedMatrix<Scalar, RowsAtCompileTime, PackingValue, 0, StructureTag>;
		TriangularCompressedBinary(Left left, Right right)
			: m_Left(std::forward<Left>(left)), m_Right(std::forward<Right>(right))
		{ eigen_assert(m_Left.dimension() == m_Right.dimension()); }
		Eigen::Index dimension() const { return m_Left.dimension(); }
		Eigen::Index rows() const { return dimension(); }
		Eigen::Index cols() const { return dimension(); }
		Eigen::Index size() const { return dimension() * dimension(); }
		Scalar coeff(Eigen::Index row, Eigen::Index column) const
		{ return applyBinary<Operation>(m_Left.coeff(row, column), m_Right.coeff(row, column)); }
		auto toDense() const { return evalDense(*this); }
		PlainObject eval() const { return PlainObject(*this); }
	private: Left m_Left; Right m_Right;
	};

	template <typename Operand, typename Factor, typename Operation, bool Structured>
	class TriangularCompressedScaled;
	template <typename Left, typename Right, typename Operation>
	class DenseCwiseBinary;
	template <typename Operand, typename Factor, typename Operation>
	class TriangularCompressedScaled<Operand, Factor, Operation, true>
	    : public Hoppy::TriangularCompressedMatrixExpr<TriangularCompressedScaled<Operand, Factor, Operation, true>>
	{
		using Source = std::remove_cv_t<std::remove_reference_t<Operand>>;
	public:
		using Scalar = binary_scalar_t<Operation, typename Source::Scalar, std::remove_cv_t<std::remove_reference_t<Factor>>>;
		using StructureTag = normalized_structure_tag_t<Scalar, typename Source::StructureTag>;
		static constexpr int RowsAtCompileTime = Source::RowsAtCompileTime;
		static constexpr int ColsAtCompileTime = Source::ColsAtCompileTime;
		static constexpr TrianglePacking PackingValue = Source::PackingValue;
		static constexpr bool IsTriangularCompressed = true;
		static constexpr bool IsWritable = false;
		using PlainObject = TriangularCompressedMatrix<Scalar, RowsAtCompileTime, PackingValue, 0, StructureTag>;
		TriangularCompressedScaled(Operand operand, Factor factor)
			: m_Operand(std::forward<Operand>(operand)), m_Factor(std::forward<Factor>(factor)) {}
		Eigen::Index dimension() const { return m_Operand.dimension(); }
		Eigen::Index rows() const { return dimension(); }
		Eigen::Index cols() const { return dimension(); }
		Eigen::Index size() const { return dimension() * dimension(); }
		Scalar coeff(Eigen::Index row, Eigen::Index column) const
		{ return applyBinary<Operation>(m_Operand.coeff(row, column), m_Factor); }
		auto toDense() const { return evalDense(*this); }
		PlainObject eval() const { return PlainObject(*this); }
	private: Operand m_Operand; Factor m_Factor;
	};


}

namespace Eigen::internal
{
	template <typename Left, typename Right, typename Operation>
	struct traits<Hoppy::Detail::DenseCwiseBinary<Left, Right, Operation>>
	{
		using Lhs = std::remove_cv_t<std::remove_reference_t<Left>>;
		using Rhs = std::remove_cv_t<std::remove_reference_t<Right>>;
		using Scalar = Hoppy::Detail::binary_scalar_t<Operation, typename Lhs::Scalar,
		                                                typename Rhs::Scalar>;
		using ReturnType = Eigen::Matrix<Scalar, Lhs::RowsAtCompileTime, Lhs::ColsAtCompileTime>;
		using StorageKind = Dense;
		using XprKind = MatrixXpr;
		using StorageIndex = Eigen::Index;
		static constexpr int Flags = EvalBeforeNestingBit;
		static constexpr int RowsAtCompileTime = Lhs::RowsAtCompileTime;
		static constexpr int ColsAtCompileTime = Lhs::ColsAtCompileTime;
		static constexpr int MaxRowsAtCompileTime = Lhs::RowsAtCompileTime;
		static constexpr int MaxColsAtCompileTime = Lhs::ColsAtCompileTime;
	};
	template <typename Operand, typename Factor, typename Operation>
	struct traits<Hoppy::Detail::TriangularCompressedScaled<Operand, Factor, Operation, false>>
	{
		using Source = std::remove_cv_t<std::remove_reference_t<Operand>>;
		using Scalar = Hoppy::Detail::binary_scalar_t<
		        Operation, typename Source::Scalar, std::remove_cv_t<std::remove_reference_t<Factor>>>;
		using ReturnType = Eigen::Matrix<Scalar, Source::RowsAtCompileTime, Source::ColsAtCompileTime>;
		using StorageKind = Dense;
		using XprKind = MatrixXpr;
		using StorageIndex = Eigen::Index;
		static constexpr int Flags = EvalBeforeNestingBit;
		static constexpr int RowsAtCompileTime = Source::RowsAtCompileTime;
		static constexpr int ColsAtCompileTime = Source::ColsAtCompileTime;
		static constexpr int MaxRowsAtCompileTime = Source::RowsAtCompileTime;
		static constexpr int MaxColsAtCompileTime = Source::ColsAtCompileTime;
	};
}

namespace Hoppy::Detail
{
	template <typename Left, typename Right, typename Operation>
	class DenseCwiseBinary
	    : public Eigen::MatrixBase<DenseCwiseBinary<Left, Right, Operation>>
	{
		using Lhs = std::remove_cv_t<std::remove_reference_t<Left>>;
		using Rhs = std::remove_cv_t<std::remove_reference_t<Right>>;
	public:
		using This = DenseCwiseBinary;
		using Base = Eigen::MatrixBase<This>;
		EIGEN_DENSE_PUBLIC_INTERFACE(This)
		using ReturnType = typename Eigen::internal::traits<DenseCwiseBinary>::ReturnType;
		static constexpr bool IsTriangularCompressed = false;
		static constexpr bool IsDensePromoted = true;
		DenseCwiseBinary(Left left, Right right) : m_Left(std::forward<Left>(left)), m_Right(std::forward<Right>(right))
		{ eigen_assert(m_Left.rows() == m_Right.rows() && m_Left.cols() == m_Right.cols()); }
		Eigen::Index rows() const { return m_Left.rows(); }
		Eigen::Index cols() const { return m_Left.cols(); }
		template <typename Destination>
		void evalTo(Destination& destination) const
		{
			decltype(auto) left = coefficientOperand(m_Left);
			decltype(auto) right = coefficientOperand(m_Right);
			ReturnType evaluated(rows(), cols());
			for (Eigen::Index row = 0; row < rows(); ++row)
				for (Eigen::Index column = 0; column < cols(); ++column)
					evaluated.coeffRef(row, column) = applyBinary<Operation>(
					        left.coeff(row, column), right.coeff(row, column));
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

	template <typename Operand, typename Factor, typename Operation>
	class TriangularCompressedScaled<Operand, Factor, Operation, false>
	    : public Eigen::MatrixBase<TriangularCompressedScaled<Operand, Factor, Operation, false>>
	{
		using Source = std::remove_cv_t<std::remove_reference_t<Operand>>;
	public:
		using This = TriangularCompressedScaled;
		using Base = Eigen::MatrixBase<This>;
		EIGEN_DENSE_PUBLIC_INTERFACE(This)
		using ReturnType = typename Eigen::internal::traits<TriangularCompressedScaled>::ReturnType;
		static constexpr bool IsTriangularCompressed = false;
		static constexpr bool IsDensePromoted = true;
		TriangularCompressedScaled(Operand operand, Factor factor)
			: m_Operand(std::forward<Operand>(operand)), m_Factor(std::forward<Factor>(factor)) {}
		Eigen::Index rows() const { return m_Operand.rows(); }
		Eigen::Index cols() const { return m_Operand.cols(); }
		template <typename Destination>
		void evalTo(Destination& destination) const
		{
			ReturnType evaluated(rows(), cols());
			for (Eigen::Index row = 0; row < rows(); ++row)
				for (Eigen::Index column = 0; column < cols(); ++column)
					evaluated.coeffRef(row, column) =
					        applyBinary<Operation>(m_Operand.coeff(row, column), m_Factor);
			destination = evaluated;
		}
		ReturnType toDense() const
		{
			ReturnType result(rows(), cols());
			evalTo(result);
			return result;
		}
	private: Operand m_Operand; Factor m_Factor;
	};

	template <typename Expression>
	auto evalDense(const Expression& expression)
	{
		Eigen::Matrix<typename Expression::Scalar, Eigen::Dynamic, Eigen::Dynamic> result(expression.rows(), expression.cols());
		for (Eigen::Index row = 0; row < expression.rows(); ++row)
			for (Eigen::Index column = 0; column < expression.cols(); ++column)
				result(row, column) = expression.coeff(row, column);
		return result;
	}
}

namespace Hoppy
{
	template <typename Factor, typename Derived,
	          typename = std::enable_if_t<Detail::is_scalar_operand_v<Factor>>>
	auto operator*(Factor&& factor, const TriangularCompressedMatrixExpr<Derived>& expression)
	{
		return expression.derived() * std::forward<Factor>(factor);
	}
	template <typename Factor, typename Derived,
	          typename = std::enable_if_t<Detail::is_scalar_operand_v<Factor>>>
	auto operator*(Factor&& factor, TriangularCompressedMatrixExpr<Derived>&& expression)
	{
		return std::move(expression.derived()) * std::forward<Factor>(factor);
	}

	template <typename Derived>
	decltype(auto) TriangularCompressedMatrixExpr<Derived>::operator+() const&
	{ return derived(); }
	template <typename Derived>
	decltype(auto) TriangularCompressedMatrixExpr<Derived>::operator+() &&
	{ return std::move(derived()); }
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::operator-() const&
	{
		return Detail::TriangularCompressedNegate<const Derived&, typename Derived::Scalar>(derived());
	}
	template <typename Derived> auto TriangularCompressedMatrixExpr<Derived>::operator-() &&
	{
		return Detail::TriangularCompressedNegate<Derived, typename Derived::Scalar>(std::move(derived()));
	}
	template <typename Derived> template <typename Other, typename>
	auto TriangularCompressedMatrixExpr<Derived>::operator+(Other&& other) const&
	{
		using Rhs = std::remove_cv_t<std::remove_reference_t<Other>>;
		using StoredRight = Detail::nested_operand_t<Other&&>;
		if constexpr (std::is_same_v<typename Derived::StructureTag, typename Rhs::StructureTag>)
			return Detail::TriangularCompressedBinary<const Derived&, StoredRight, Detail::SumOperation>(derived(), std::forward<Other>(other));
		else return Detail::DenseCwiseBinary<const Derived&, StoredRight, Detail::SumOperation>(derived(), std::forward<Other>(other));
	}
	template <typename Derived> template <typename Other, typename>
	auto TriangularCompressedMatrixExpr<Derived>::operator+(Other&& other) &&
	{
		using Rhs = std::remove_cv_t<std::remove_reference_t<Other>>;
		using StoredRight = Detail::nested_operand_t<Other&&>;
		if constexpr (std::is_same_v<typename Derived::StructureTag, typename Rhs::StructureTag>)
			return Detail::TriangularCompressedBinary<Derived, StoredRight, Detail::SumOperation>(std::move(derived()), std::forward<Other>(other));
		else return Detail::DenseCwiseBinary<Derived, StoredRight, Detail::SumOperation>(std::move(derived()), std::forward<Other>(other));
	}
	template <typename Derived> template <typename Other, typename>
	auto TriangularCompressedMatrixExpr<Derived>::operator-(Other&& other) const&
	{
		using Rhs = std::remove_cv_t<std::remove_reference_t<Other>>;
		using StoredRight = Detail::nested_operand_t<Other&&>;
		if constexpr (std::is_same_v<typename Derived::StructureTag, typename Rhs::StructureTag>)
			return Detail::TriangularCompressedBinary<const Derived&, StoredRight, Detail::DifferenceOperation>(derived(), std::forward<Other>(other));
		else return Detail::DenseCwiseBinary<const Derived&, StoredRight, Detail::DifferenceOperation>(derived(), std::forward<Other>(other));
	}
	template <typename Derived> template <typename Other, typename>
	auto TriangularCompressedMatrixExpr<Derived>::operator-(Other&& other) &&
	{
		using Rhs = std::remove_cv_t<std::remove_reference_t<Other>>;
		using StoredRight = Detail::nested_operand_t<Other&&>;
		if constexpr (std::is_same_v<typename Derived::StructureTag, typename Rhs::StructureTag>)
			return Detail::TriangularCompressedBinary<Derived, StoredRight, Detail::DifferenceOperation>(std::move(derived()), std::forward<Other>(other));
		else return Detail::DenseCwiseBinary<Derived, StoredRight, Detail::DifferenceOperation>(std::move(derived()), std::forward<Other>(other));
	}
	template <typename Derived> template <typename Factor, typename>
	auto TriangularCompressedMatrixExpr<Derived>::operator*(Factor&& factor) const&
	{
		using StoredFactor = std::remove_cv_t<std::remove_reference_t<Factor>>;
		return Detail::TriangularCompressedScaled<const Derived&, StoredFactor, Detail::ProductOperation,
		        Detail::preserves_scaled_structure_v<Derived, StoredFactor>>(derived(), std::forward<Factor>(factor));
	}
	template <typename Derived> template <typename Factor, typename>
	auto TriangularCompressedMatrixExpr<Derived>::operator*(Factor&& factor) &&
	{
		using StoredFactor = std::remove_cv_t<std::remove_reference_t<Factor>>;
		return Detail::TriangularCompressedScaled<Derived, StoredFactor, Detail::ProductOperation,
		        Detail::preserves_scaled_structure_v<Derived, StoredFactor>>(std::move(derived()), std::forward<Factor>(factor));
	}
	template <typename Derived> template <typename Divisor>
	auto TriangularCompressedMatrixExpr<Derived>::operator/(Divisor&& divisor) const&
	{
		using StoredDivisor = std::remove_cv_t<std::remove_reference_t<Divisor>>;
		return Detail::TriangularCompressedScaled<const Derived&, StoredDivisor, Detail::QuotientOperation,
		        Detail::preserves_scaled_structure_v<Derived, StoredDivisor>>(derived(), std::forward<Divisor>(divisor));
	}
	template <typename Derived> template <typename Divisor>
	auto TriangularCompressedMatrixExpr<Derived>::operator/(Divisor&& divisor) &&
	{
		using StoredDivisor = std::remove_cv_t<std::remove_reference_t<Divisor>>;
		return Detail::TriangularCompressedScaled<Derived, StoredDivisor, Detail::QuotientOperation,
		        Detail::preserves_scaled_structure_v<Derived, StoredDivisor>>(std::move(derived()), std::forward<Divisor>(divisor));
	}
	template <typename Derived> template <typename NewScalar>
	auto TriangularCompressedMatrixExpr<Derived>::cast() const&
	{ return Detail::TriangularCompressedCast<const Derived&, NewScalar>(derived()); }
	template <typename Derived> template <typename NewScalar>
	auto TriangularCompressedMatrixExpr<Derived>::cast() &&
	{ return Detail::TriangularCompressedCast<Derived, NewScalar>(std::move(derived())); }

	template <typename Left, typename Dense,
	          typename = std::enable_if_t<Detail::is_triangular_expression<
	                  std::remove_cv_t<std::remove_reference_t<Left>>>::value
	                                      && Detail::is_dense_matrix_operand_v<Dense>>>
	auto operator+(Left&& left, Dense&& dense)
	{
		using StoredLeft = Detail::nested_operand_t<Left&&>;
		using StoredRight = Detail::nested_operand_t<Dense&&>;
		return Detail::DenseCwiseBinary<StoredLeft, StoredRight, Detail::SumOperation>(
		        std::forward<Left>(left), std::forward<Dense>(dense));
	}
	template <typename Dense, typename Right,
	          typename = std::enable_if_t<Detail::is_dense_matrix_operand_v<Dense>
	                                      && Detail::is_triangular_expression<
	                                              std::remove_cv_t<std::remove_reference_t<Right>>>::value>,
	          typename = void>
	auto operator+(Dense&& dense, Right&& right)
	{
		using StoredLeft = Detail::nested_operand_t<Dense&&>;
		using StoredRight = Detail::nested_operand_t<Right&&>;
		return Detail::DenseCwiseBinary<StoredLeft, StoredRight, Detail::SumOperation>(
		        std::forward<Dense>(dense), std::forward<Right>(right));
	}
	template <typename Left, typename Dense,
	          typename = std::enable_if_t<Detail::is_triangular_expression<
	                  std::remove_cv_t<std::remove_reference_t<Left>>>::value
	                                      && Detail::is_dense_matrix_operand_v<Dense>>>
	auto operator-(Left&& left, Dense&& dense)
	{
		using StoredLeft = Detail::nested_operand_t<Left&&>;
		using StoredRight = Detail::nested_operand_t<Dense&&>;
		return Detail::DenseCwiseBinary<StoredLeft, StoredRight, Detail::DifferenceOperation>(
		        std::forward<Left>(left), std::forward<Dense>(dense));
	}
	template <typename Dense, typename Right,
	          typename = std::enable_if_t<Detail::is_dense_matrix_operand_v<Dense>
	                                      && Detail::is_triangular_expression<
	                                              std::remove_cv_t<std::remove_reference_t<Right>>>::value>,
	          typename = void>
	auto operator-(Dense&& dense, Right&& right)
	{
		using StoredLeft = Detail::nested_operand_t<Dense&&>;
		using StoredRight = Detail::nested_operand_t<Right&&>;
		return Detail::DenseCwiseBinary<StoredLeft, StoredRight, Detail::DifferenceOperation>(
		        std::forward<Dense>(dense), std::forward<Right>(right));
	}

	template <typename Derived>
	auto TriangularCompressedMatrixExpr<Derived>::transpose() const&
	{
		return Detail::TriangularCompressedTransform<const Derived&, Detail::TransposeOperation>(derived());
	}
	template <typename Derived>
	auto TriangularCompressedMatrixExpr<Derived>::transpose() &&
	{
		return Detail::TriangularCompressedTransform<Derived, Detail::TransposeOperation>(std::move(derived()));
	}
	template <typename Derived>
	auto TriangularCompressedMatrixExpr<Derived>::conjugate() const&
	{
		return Detail::TriangularCompressedTransform<const Derived&, Detail::ConjugateOperation>(derived());
	}
	template <typename Derived>
	auto TriangularCompressedMatrixExpr<Derived>::conjugate() &&
	{
		return Detail::TriangularCompressedTransform<Derived, Detail::ConjugateOperation>(std::move(derived()));
	}
	template <typename Derived>
	auto TriangularCompressedMatrixExpr<Derived>::adjoint() const&
	{
		return Detail::TriangularCompressedTransform<const Derived&, Detail::AdjointOperation>(derived());
	}
	template <typename Derived>
	auto TriangularCompressedMatrixExpr<Derived>::adjoint() &&
	{
		return Detail::TriangularCompressedTransform<Derived, Detail::AdjointOperation>(std::move(derived()));
	}
}

namespace Eigen::internal
{
	template <typename Expression>
	struct triangular_dense_evaluator
	    : evaluator<typename traits<Expression>::ReturnType>
	{
		using ReturnType = typename traits<Expression>::ReturnType;
		using Base = evaluator<ReturnType>;
		explicit triangular_dense_evaluator(const Expression& expression)
		    : m_Result(expression.rows(), expression.cols())
		{
			Eigen::internal::construct_at<Base>(this, m_Result);
			expression.evalTo(m_Result);
		}
	protected:
		ReturnType m_Result;
	};

	template <typename Left, typename Right, typename Operation>
	struct evaluator<Hoppy::Detail::DenseCwiseBinary<Left, Right, Operation>>
	    : triangular_dense_evaluator<Hoppy::Detail::DenseCwiseBinary<Left, Right, Operation>>
	{
		using Expression = Hoppy::Detail::DenseCwiseBinary<Left, Right, Operation>;
		using Base = triangular_dense_evaluator<Expression>;
		explicit evaluator(const Expression& expression) : Base(expression) {}
	};
	template <typename Operand, typename Factor, typename Operation>
	struct evaluator<Hoppy::Detail::TriangularCompressedScaled<Operand, Factor, Operation, false>>
	    : triangular_dense_evaluator<
	              Hoppy::Detail::TriangularCompressedScaled<Operand, Factor, Operation, false>>
	{
		using Expression = Hoppy::Detail::TriangularCompressedScaled<Operand, Factor, Operation, false>;
		using Base = triangular_dense_evaluator<Expression>;
		explicit evaluator(const Expression& expression) : Base(expression) {}
	};

	template <typename Expression>
	struct triangular_node_traits
	{
		using Scalar = typename Expression::Scalar;
		using StorageKind = Hoppy::Detail::TriangularCompressedStorage;
		using XprKind = MatrixXpr;
		using StorageIndex = Eigen::Index;
		static constexpr int Flags = 0;
		static constexpr int RowsAtCompileTime = Expression::RowsAtCompileTime;
		static constexpr int ColsAtCompileTime = Expression::ColsAtCompileTime;
		static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
		static constexpr int MaxColsAtCompileTime = ColsAtCompileTime;
	};

	template <typename Operand, typename Scalar>
	struct traits<Hoppy::Detail::TriangularCompressedNegate<Operand, Scalar>>
	    : triangular_node_traits<Hoppy::Detail::TriangularCompressedNegate<Operand, Scalar>> {};
	template <typename Operand, typename Scalar>
	struct traits<Hoppy::Detail::TriangularCompressedCast<Operand, Scalar>>
	    : triangular_node_traits<Hoppy::Detail::TriangularCompressedCast<Operand, Scalar>> {};
	template <typename Left, typename Right, typename Operation>
	struct traits<Hoppy::Detail::TriangularCompressedBinary<Left, Right, Operation>>
	    : triangular_node_traits<Hoppy::Detail::TriangularCompressedBinary<Left, Right, Operation>> {};
	template <typename Operand, typename Factor, typename Operation>
	struct traits<Hoppy::Detail::TriangularCompressedScaled<Operand, Factor, Operation, true>>
	    : triangular_node_traits<Hoppy::Detail::TriangularCompressedScaled<Operand, Factor, Operation, true>> {};

	template <typename Operand, typename Scalar>
	struct evaluator<Hoppy::Detail::TriangularCompressedNegate<Operand, Scalar>>
	    : triangular_compressed_evaluator<Hoppy::Detail::TriangularCompressedNegate<Operand, Scalar>>
	{
		using Expression = Hoppy::Detail::TriangularCompressedNegate<Operand, Scalar>;
		using Base = triangular_compressed_evaluator<Expression>;
		explicit evaluator(const Expression& expression) : Base(expression) {}
	};
	template <typename Operand, typename Scalar>
	struct evaluator<Hoppy::Detail::TriangularCompressedCast<Operand, Scalar>>
	    : triangular_compressed_evaluator<Hoppy::Detail::TriangularCompressedCast<Operand, Scalar>>
	{
		using Expression = Hoppy::Detail::TriangularCompressedCast<Operand, Scalar>;
		using Base = triangular_compressed_evaluator<Expression>;
		explicit evaluator(const Expression& expression) : Base(expression) {}
	};
	template <typename Left, typename Right, typename Operation>
	struct evaluator<Hoppy::Detail::TriangularCompressedBinary<Left, Right, Operation>>
	    : triangular_compressed_evaluator<Hoppy::Detail::TriangularCompressedBinary<Left, Right, Operation>>
	{
		using Expression = Hoppy::Detail::TriangularCompressedBinary<Left, Right, Operation>;
		using Base = triangular_compressed_evaluator<Expression>;
		explicit evaluator(const Expression& expression) : Base(expression) {}
	};
	template <typename Operand, typename Factor, typename Operation>
	struct evaluator<Hoppy::Detail::TriangularCompressedScaled<Operand, Factor, Operation, true>>
	    : triangular_compressed_evaluator<Hoppy::Detail::TriangularCompressedScaled<Operand, Factor, Operation, true>>
	{
		using Expression = Hoppy::Detail::TriangularCompressedScaled<Operand, Factor, Operation, true>;
		using Base = triangular_compressed_evaluator<Expression>;
		explicit evaluator(const Expression& expression) : Base(expression) {}
	};

	template <typename Operand, typename Operation>
	struct traits<Hoppy::Detail::TriangularCompressedTransform<Operand, Operation>>
	{
		using Expression = Hoppy::Detail::TriangularCompressedTransform<Operand, Operation>;
		using Scalar = typename Expression::Scalar;
		using StorageKind = Hoppy::Detail::TriangularCompressedStorage;
		using XprKind = MatrixXpr;
		using StorageIndex = Eigen::Index;
		static constexpr int Flags = 0;
		static constexpr int RowsAtCompileTime = Expression::RowsAtCompileTime;
		static constexpr int ColsAtCompileTime = Expression::ColsAtCompileTime;
		static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
		static constexpr int MaxColsAtCompileTime = ColsAtCompileTime;
	};

	template <typename Operand, typename Operation>
	struct evaluator<Hoppy::Detail::TriangularCompressedTransform<Operand, Operation>>
	    : triangular_compressed_evaluator<
	              Hoppy::Detail::TriangularCompressedTransform<Operand, Operation>>
	{
		using Expression = Hoppy::Detail::TriangularCompressedTransform<Operand, Operation>;
		using Base = triangular_compressed_evaluator<Expression>;
		explicit evaluator(const Expression& expression) : Base(expression) {}
	};
}
