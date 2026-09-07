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
}  // namespace Eigen::internal
