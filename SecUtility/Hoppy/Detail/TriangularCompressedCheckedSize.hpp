// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/TrianglePacking.hpp>

#include <Eigen/Core>

#include <cstddef>
#include <limits>
#include <utility>

namespace Hoppy::Detail
{
	inline Eigen::Index checkedDimension(const std::size_t dimension)
	{
		static_assert(std::numeric_limits<std::size_t>::digits >= std::numeric_limits<Eigen::Index>::digits);
		if (dimension > static_cast<std::size_t>((std::numeric_limits<Eigen::Index>::max)()))
		{
			eigen_assert(false && "triangular-compressed dimension is not representable as Eigen::Index");
			return 0;
		}
		return static_cast<Eigen::Index>(dimension);
	}

	inline Eigen::Index checkedSquareDimension(const Eigen::Index rows, const Eigen::Index columns)
	{
		if (rows < 0 || columns < 0 || rows != columns)
		{
			eigen_assert(false && "triangular-compressed dimensions must be nonnegative and square");
			return 0;
		}
		return rows;
	}

	inline Eigen::Index checkedLogicalSize(const Eigen::Index dimension)
	{
		if (dimension < 0)
		{
			eigen_assert(false && "triangular-compressed dimension must be nonnegative");
			return 0;
		}
		if (dimension != 0 && dimension > std::numeric_limits<Eigen::Index>::max() / dimension)
		{
			eigen_assert(false && "triangular-compressed logical size overflows Eigen::Index");
			return 0;
		}
		return dimension * dimension;
	}

	inline Eigen::Index checkedStoredSize(const Eigen::Index dimension)
	{
		if (dimension < 0)
		{
			eigen_assert(false && "triangular-compressed dimension must be nonnegative");
			return 0;
		}
		const Eigen::Index left = dimension % 2 == 0 ? dimension / 2 : dimension;
		const Eigen::Index right = dimension % 2 == 0 ? dimension + 1 : dimension / 2 + 1;
		if (left != 0 && right > std::numeric_limits<Eigen::Index>::max() / left)
		{
			eigen_assert(false && "triangular-compressed stored size overflows Eigen::Index");
			return 0;
		}
		return left * right;
	}

	inline std::size_t checkedStoredByteCount(const Eigen::Index dimension, const std::size_t scalarSize)
	{
		const Eigen::Index storedSize = checkedStoredSize(dimension);
		if (scalarSize != 0
		    && static_cast<std::size_t>(storedSize) > std::numeric_limits<std::size_t>::max() / scalarSize)
		{
			eigen_assert(false && "triangular-compressed byte count overflows size_t");
			return 0;
		}
		return static_cast<std::size_t>(storedSize) * scalarSize;
	}

	template <typename Scalar>
	std::size_t checkedStoredByteCount(const Eigen::Index dimension)
	{
		return checkedStoredByteCount(dimension, sizeof(Scalar));
	}

	inline std::pair<Eigen::Index, Eigen::Index> canonicalPackedCoordinate(Eigen::Index row,
	                                                                       Eigen::Index column,
	                                                                       const TrianglePacking packing)
	{
		if (packing != TrianglePacking::Lower && packing != TrianglePacking::Upper)
		{
			eigen_assert(false && "invalid triangle packing");
			return {0, 0};
		}
		if (packing == TrianglePacking::Lower)
		{
			return row >= column ? std::make_pair(row, column) : std::make_pair(column, row);
		}
		return row <= column ? std::make_pair(row, column) : std::make_pair(column, row);
	}

	inline Eigen::Index checkedPackedOffset(const Eigen::Index row,
	                                        const Eigen::Index column,
	                                        const Eigen::Index dimension,
	                                        const TrianglePacking packing)
	{
		if ((packing != TrianglePacking::Lower && packing != TrianglePacking::Upper) || dimension < 0 || row < 0
		    || column < 0 || row >= dimension || column >= dimension)
		{
			eigen_assert(false && "triangular-compressed coefficient index is out of bounds");
			return 0;
		}
		const auto canonical = canonicalPackedCoordinate(row, column, packing);
		const Eigen::Index major = packing == TrianglePacking::Lower ? canonical.first : canonical.second;
		const Eigen::Index minor = packing == TrianglePacking::Lower ? canonical.second : canonical.first;
		const Eigen::Index prefix = checkedStoredSize(major);
		if (minor > std::numeric_limits<Eigen::Index>::max() - prefix)
		{
			eigen_assert(false && "triangular-compressed coefficient offset overflows Eigen::Index");
			return 0;
		}
		return prefix + minor;
	}
}  // namespace Hoppy::Detail
