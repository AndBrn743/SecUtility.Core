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
		static constexpr int Flags = Eigen::NestByRefBit;
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
}

namespace Hoppy
{
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
	template <typename Operand, typename Operation>
	struct traits<Hoppy::Detail::TriangularCompressedTransform<Operand, Operation>>
	{
		using Expression = Hoppy::Detail::TriangularCompressedTransform<Operand, Operation>;
		using Scalar = typename Expression::Scalar;
		using StorageKind = Hoppy::Detail::TriangularCompressedStorage;
		using XprKind = MatrixXpr;
		using StorageIndex = Eigen::Index;
		static constexpr int Flags = NestByRefBit;
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
