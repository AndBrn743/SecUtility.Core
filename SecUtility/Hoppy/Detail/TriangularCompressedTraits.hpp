// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>

#include <Eigen/Core>

namespace Hoppy::Detail
{
	struct TriangularCompressedStorage
	{};
	struct TriangularCompressedShape
	{};
}  // namespace Hoppy::Detail

namespace Eigen::internal
{
	template <typename Expression>
	struct triangular_compressed_evaluator : evaluator_base<Expression>
	{
		using Scalar = typename traits<Expression>::Scalar;
		enum { CoeffReadCost = NumTraits<Scalar>::ReadCost, Flags = 0, Alignment = 0 };
		explicit triangular_compressed_evaluator(const Expression& expression) : m_Expression(expression) {}
		Scalar coeff(Eigen::Index row, Eigen::Index column) const
		{
			return m_Expression.coeff(row, column);
		}
		const Expression& m_Expression;
	};

	template <>
	struct storage_kind_to_shape<Hoppy::Detail::TriangularCompressedStorage>
	{
		using Shape = Hoppy::Detail::TriangularCompressedShape;
	};

	template <typename TScalar, int Dimension, Hoppy::TrianglePacking Packing, int Options,
	          typename StructureTag>
	struct traits<Hoppy::Detail::TriangularCompressedMatrix<
	        TScalar, Dimension, Packing, Options, StructureTag>>
	{
		using Scalar = TScalar;
		using StorageKind = Hoppy::Detail::TriangularCompressedStorage;
		using XprKind = MatrixXpr;
		using StorageIndex = Eigen::Index;
		static constexpr int Flags = NestByRefBit;
		static constexpr int RowsAtCompileTime = Dimension;
		static constexpr int ColsAtCompileTime = Dimension;
		static constexpr int MaxRowsAtCompileTime = Dimension;
		static constexpr int MaxColsAtCompileTime = Dimension;
	};

	template <typename TScalar, int Dimension, Hoppy::TrianglePacking Packing, int Options,
	          typename StructureTag>
	struct evaluator<Hoppy::Detail::TriangularCompressedMatrix<
	        TScalar, Dimension, Packing, Options, StructureTag>>
	    : triangular_compressed_evaluator<Hoppy::Detail::TriangularCompressedMatrix<
	              TScalar, Dimension, Packing, Options, StructureTag>>
	{
		using Expression = Hoppy::Detail::TriangularCompressedMatrix<
		        TScalar, Dimension, Packing, Options, StructureTag>;
		using Base = triangular_compressed_evaluator<Expression>;
		explicit evaluator(const Expression& expression) : Base(expression) {}
	};
}  // namespace Eigen::internal
