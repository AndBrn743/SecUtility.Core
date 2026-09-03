// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>

#include <Eigen/Core>

#include <limits>
#include <type_traits>

namespace Hoppy
{
	struct DenseBlockPolicy
	{
		static Eigen::Index BlockElementCount(const Eigen::Index dimension)
		{
			if (dimension <= 0 || dimension > std::numeric_limits<Eigen::Index>::max() / dimension)
			{
				eigen_assert(false && "dense block dimension must be positive and square without overflow");
				return 0;
			}
			return dimension * dimension;
		}

		template <typename TScalar>
		using View = Eigen::Map<Eigen::MatrixX<TScalar>, Eigen::Unaligned>;

		template <typename TScalar>
		using ConstView = Eigen::Map<const Eigen::MatrixX<TScalar>, Eigen::Unaligned>;
	};

	template <typename T>
	struct is_block_policy : std::false_type
	{};

	template <>
	struct is_block_policy<DenseBlockPolicy> : std::true_type
	{};

	template <typename T>
	inline constexpr bool IsBlockPolicy = is_block_policy<T>::value;
}  // namespace Hoppy
