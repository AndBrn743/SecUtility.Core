// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/TrianglePacking.hpp>

#include <type_traits>
#include <limits>

namespace Hoppy::Detail
{
	struct UpperTriangularTag
	{};
	struct LowerTriangularTag
	{};
	struct SymmetricTag
	{};
	struct AntiSymmetricTag
	{};
	struct HermitianTag
	{};
	struct AntiHermitianTag
	{};

	template <typename Scalar, typename StructureTag>
	using normalized_structure_tag_t = std::conditional_t<
	        Eigen::NumTraits<Scalar>::IsComplex == 0,
	        std::conditional_t<std::is_same_v<StructureTag, HermitianTag>, SymmetricTag,
	                           std::conditional_t<std::is_same_v<StructureTag, AntiHermitianTag>,
	                                              AntiSymmetricTag, StructureTag>>,
	        StructureTag>;

	template <int Dimension>
	inline constexpr bool is_valid_triangular_dimension_v = Dimension == Eigen::Dynamic || Dimension >= 0;

	template <int Dimension>
	inline constexpr bool has_representable_logical_size_v = Dimension == Eigen::Dynamic || Dimension == 0
	                                                        || Dimension <= (std::numeric_limits<int>::max)()
	                                                                                / Dimension;

	template <int Dimension>
	constexpr int compileTimeLogicalSize()
	{
		if constexpr (Dimension == Eigen::Dynamic)
			return Eigen::Dynamic;
		else if constexpr (has_representable_logical_size_v<Dimension>)
			return Dimension * Dimension;
		else
			return Eigen::Dynamic;
	}

	template <TrianglePacking Packing>
	inline constexpr bool is_valid_triangle_packing_v = Packing == TrianglePacking::Lower
	                                                   || Packing == TrianglePacking::Upper;

	// Eigen storage options are bit fields: bit 0 is RowMajor and bit 1 is DontAlign.
	// Thus 0 means ColMajor/AutoAlign, and the only other accepted value is DontAlign.
	template <int Options>
	inline constexpr bool is_valid_triangular_options_v = (Options & ~Eigen::DontAlign) == 0;
}  // namespace Hoppy::Detail
