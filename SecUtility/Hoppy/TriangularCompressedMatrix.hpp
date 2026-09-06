// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedCheckedSize.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedTags.hpp>

#include <Eigen/Core>

namespace Hoppy::Detail
{
	template <typename TScalar, int Dimension, TrianglePacking Packing, int Options,
	          typename NormalizedStructureTag>
	class TriangularCompressedMatrix
	{
		static_assert(is_valid_triangular_dimension_v<Dimension>,
		              "Dimension must be nonnegative or Eigen::Dynamic");
		static_assert(has_representable_logical_size_v<Dimension>,
		              "fixed logical coefficient count must be representable as int");
		static_assert(is_valid_triangle_packing_v<Packing>, "Packing must be Lower or Upper");
		static_assert(is_valid_triangular_options_v<Options>,
		              "Options may contain only Eigen::DontAlign; row-major and unrelated bits are unsupported");

	public:
		using Scalar = TScalar;
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		using StorageIndex = Eigen::Index;
		using StructureTag = NormalizedStructureTag;
		using PlainObject = TriangularCompressedMatrix;
		using Nested = const TriangularCompressedMatrix&;

		static constexpr int RowsAtCompileTime = Dimension;
		static constexpr int ColsAtCompileTime = Dimension;
		static constexpr int MaxRowsAtCompileTime = Dimension;
		static constexpr int MaxColsAtCompileTime = Dimension;
		static constexpr int SizeAtCompileTime = compileTimeLogicalSize<Dimension>();
		static constexpr int MaxSizeAtCompileTime = SizeAtCompileTime;
		static constexpr int Flags = Eigen::NestByRefBit;
		static constexpr TrianglePacking PackingValue = Packing;

		static Eigen::Index requiredStoredSize(const Eigen::Index dimension)
		{
			if constexpr (Dimension != Eigen::Dynamic)
			{
				if (dimension != Dimension)
				{
					eigen_assert(false && "runtime dimension does not match fixed dimension");
					return 0;
				}
			}
			return checkedStoredSize(dimension);
		}
	};
}  // namespace Hoppy::Detail
