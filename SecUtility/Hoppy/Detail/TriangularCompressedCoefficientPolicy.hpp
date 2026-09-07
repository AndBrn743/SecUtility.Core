// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/Detail/TriangularCompressedCheckedSize.hpp>
#include <SecUtility/Hoppy/Detail/TriangularCompressedTags.hpp>

#include <Eigen/Core>

#include <type_traits>

namespace Hoppy::Detail
{
	template <typename Scalar, typename StructureTag, TrianglePacking Packing>
	struct TriangularCompressedCoefficientPolicy
	{
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

		static bool isValidIndex(const Eigen::Index dimension,
		                         const Eigen::Index row,
		                         const Eigen::Index column) noexcept
		{
			return row >= 0 && column >= 0 && row < dimension && column < dimension;
		}

		static bool isCanonicalSide(const Eigen::Index row, const Eigen::Index column) noexcept
		{
			return Packing == TrianglePacking::Lower ? row >= column : row <= column;
		}

		static bool isAuthoritativeCoordinate(const Eigen::Index row, const Eigen::Index column) noexcept
		{
			if constexpr (std::is_same_v<StructureTag, UpperTriangularTag>) return row <= column;
			else if constexpr (std::is_same_v<StructureTag, LowerTriangularTag>) return row >= column;
			else return isCanonicalSide(row, column);
		}

		static bool isStructuralZero(const Eigen::Index row, const Eigen::Index column) noexcept
		{
			if constexpr (std::is_same_v<StructureTag, UpperTriangularTag>) return row > column;
			else if constexpr (std::is_same_v<StructureTag, LowerTriangularTag>) return row < column;
			else return false;
		}

		static bool isValidDiagonal(const Scalar& value)
		{
			if constexpr (std::is_same_v<StructureTag, AntiSymmetricTag>) return value == Scalar(0);
			else if constexpr (std::is_same_v<StructureTag, HermitianTag>)
				return Eigen::numext::imag(value) == RealScalar(0);
			else if constexpr (std::is_same_v<StructureTag, AntiHermitianTag>)
				return Eigen::numext::real(value) == RealScalar(0);
			else return true;
		}

		static Scalar canonicalizeDiagonal(const Scalar& value)
		{
			if constexpr (std::is_same_v<StructureTag, AntiSymmetricTag>) return Scalar(0);
			else if constexpr (std::is_same_v<StructureTag, HermitianTag>)
				return Scalar(Eigen::numext::real(value));
			else if constexpr (std::is_same_v<StructureTag, AntiHermitianTag>)
				return Scalar(0, Eigen::numext::imag(value));
			else return value;
		}

		static Scalar storedToLogical(const Scalar& stored, const Eigen::Index row, const Eigen::Index column)
		{
			if (row == column || isCanonicalSide(row, column)) return stored;
			if constexpr (std::is_same_v<StructureTag, AntiSymmetricTag>) return -stored;
			else if constexpr (std::is_same_v<StructureTag, HermitianTag>) return Eigen::numext::conj(stored);
			else if constexpr (std::is_same_v<StructureTag, AntiHermitianTag>) return -Eigen::numext::conj(stored);
			else return stored;
		}

		static Scalar logicalToStored(const Scalar& logical, const Eigen::Index row, const Eigen::Index column)
		{
			if (row == column || isCanonicalSide(row, column)) return logical;
			if constexpr (std::is_same_v<StructureTag, AntiSymmetricTag>) return -logical;
			else if constexpr (std::is_same_v<StructureTag, HermitianTag>) return Eigen::numext::conj(logical);
			else if constexpr (std::is_same_v<StructureTag, AntiHermitianTag>) return -Eigen::numext::conj(logical);
			else return logical;
		}

		static Eigen::Index offset(const Eigen::Index dimension, const Eigen::Index row, const Eigen::Index column)
		{
			return checkedPackedOffset(row, column, dimension, Packing);
		}
	};
}  // namespace Hoppy::Detail
