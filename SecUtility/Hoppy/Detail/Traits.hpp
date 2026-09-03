// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>

#include <Eigen/Core>

#include <type_traits>

namespace Hoppy::Detail
{
	struct BlockDiagonalStorage
	{};
	struct BlockVectorStorage
	{};
	struct BlockDiagonalShape
	{};
	struct BlockVectorShape
	{};
}  // namespace Hoppy::Detail

namespace Eigen::internal
{
	template <>
	struct storage_kind_to_shape<Hoppy::Detail::BlockDiagonalStorage>
	{
		using Shape = Hoppy::Detail::BlockDiagonalShape;
	};

	template <>
	struct storage_kind_to_shape<Hoppy::Detail::BlockVectorStorage>
	{
		using Shape = Hoppy::Detail::BlockVectorShape;
	};

	template <typename TScalar, typename TBlockPolicy>
	struct traits<Hoppy::BlockDiagonalMatrix<TScalar, TBlockPolicy>>
	{
		using Scalar = TScalar;
		using BlockPolicy = TBlockPolicy;
		using StorageKind = Hoppy::Detail::BlockDiagonalStorage;
		using XprKind = MatrixXpr;
		using StorageIndex = Eigen::Index;
		static constexpr int Flags = NestByRefBit;
		static constexpr int RowsAtCompileTime = Dynamic;
		static constexpr int ColsAtCompileTime = Dynamic;
		static constexpr int MaxRowsAtCompileTime = Dynamic;
		static constexpr int MaxColsAtCompileTime = Dynamic;
	};

	template <typename TScalar, typename TOrientation>
	struct traits<Hoppy::BlockVector<TScalar, TOrientation>>
	{
		using Scalar = TScalar;
		using Orientation = TOrientation;
		using StorageKind = Hoppy::Detail::BlockVectorStorage;
		using XprKind = MatrixXpr;
		using StorageIndex = Eigen::Index;
		static constexpr int Flags = NestByRefBit;
		static constexpr int RowsAtCompileTime = std::is_same_v<TOrientation, Hoppy::Row> ? 1 : Dynamic;
		static constexpr int ColsAtCompileTime = std::is_same_v<TOrientation, Hoppy::Row> ? Dynamic : 1;
		static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime;
		static constexpr int MaxColsAtCompileTime = ColsAtCompileTime;
	};
}  // namespace Eigen::internal
