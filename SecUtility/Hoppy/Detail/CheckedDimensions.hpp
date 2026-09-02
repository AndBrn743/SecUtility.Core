// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/BlockPolicy.hpp>
#include <SecUtility/Hoppy/ForwardDeclarations.hpp>

#include <Eigen/Core>

#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace Hoppy::Detail
{
	struct CheckedDimensions
	{
		std::vector<Eigen::Index> Dimensions;
		std::vector<Eigen::Index> BlockOffsets{0};
		std::vector<Eigen::Index> StorageOffsets{0};
		Eigen::Index TotalDimension = 0;
		Eigen::Index StoredSize = 0;
	};

	namespace CheckedDimensionsDetail
	{
		struct DenseStorageRule
		{
			static bool tryElementCount(Eigen::Index dimension, Eigen::Index& result);
			static constexpr bool CheckRepresentedSquare = true;
		};

		struct VectorStorageRule
		{
			static bool tryElementCount(Eigen::Index dimension, Eigen::Index& result)
			{
				result = dimension;
				return true;
			}
			static constexpr bool CheckRepresentedSquare = false;
		};

		template <typename Iterator>
		using Value = std::remove_cv_t<typename std::iterator_traits<Iterator>::value_type>;

		template <typename Iterator, typename = void>
		struct is_supported_iterator : std::false_type
		{};

		template <typename Iterator>
		struct is_supported_iterator<
		        Iterator,
		        std::void_t<typename std::iterator_traits<Iterator>::iterator_category,
		                    typename std::iterator_traits<Iterator>::value_type>>
		    : std::bool_constant<
		              std::is_base_of_v<std::forward_iterator_tag,
		                                typename std::iterator_traits<Iterator>::iterator_category>
		              && std::is_integral_v<Value<Iterator>> && !std::is_same_v<Value<Iterator>, bool>>
		{};

		template <typename TInteger>
		bool tryConvertPositive(TInteger value, Eigen::Index& result)
		{
			static_assert(std::is_integral_v<TInteger> && !std::is_same_v<TInteger, bool>);
			if constexpr (std::is_signed_v<TInteger>)
			{
				if (value <= 0)
					return false;
			}
			else if (value == 0)
				return false;

			using UnsignedInput = std::make_unsigned_t<TInteger>;
			using UnsignedIndex = std::make_unsigned_t<Eigen::Index>;
			const auto unsignedValue = static_cast<UnsignedInput>(value);
			constexpr auto maximumIndex = static_cast<UnsignedIndex>(std::numeric_limits<Eigen::Index>::max());
			if constexpr (sizeof(UnsignedInput) > sizeof(UnsignedIndex))
			{
				if (unsignedValue > static_cast<UnsignedInput>(maximumIndex))
					return false;
			}
			else if (static_cast<UnsignedIndex>(unsignedValue) > maximumIndex)
				return false;

			result = static_cast<Eigen::Index>(value);
			return true;
		}

		inline bool checkedAdd(Eigen::Index lhs, Eigen::Index rhs, Eigen::Index& result)
		{
			const auto maximum = std::numeric_limits<Eigen::Index>::max();
			if (rhs > maximum - lhs)
				return false;
			result = lhs + rhs;
			return true;
		}

		inline bool checkedSquare(Eigen::Index value, Eigen::Index& result)
		{
			const auto maximum = std::numeric_limits<Eigen::Index>::max();
			if (value > maximum / value)
				return false;
			result = value * value;
			return true;
		}

		inline bool DenseStorageRule::tryElementCount(Eigen::Index dimension, Eigen::Index& result)
		{
			return checkedSquare(dimension, result);
		}

		inline CheckedDimensions rejectInvalidDimensions()
		{
			eigen_assert(false && "Hoppy blocking dimensions must be positive, representable, and overflow-free");
			return {};
		}
	}  // namespace CheckedDimensionsDetail

	template <typename StorageRule, typename Iterator,
	          std::enable_if_t<CheckedDimensionsDetail::is_supported_iterator<Iterator>::value, int> = 0>
	CheckedDimensions BuildCheckedDimensionsWith(Iterator first, Iterator last)
	{
		CheckedDimensions result;
		for (; first != last; ++first)
		{
			const auto inputDimension = *first;
			Eigen::Index dimension = 0;
			if (!CheckedDimensionsDetail::tryConvertPositive(inputDimension, dimension))
				return CheckedDimensionsDetail::rejectInvalidDimensions();

			Eigen::Index blockElements = 0;
			if (!StorageRule::tryElementCount(dimension, blockElements))
				return CheckedDimensionsDetail::rejectInvalidDimensions();

			Eigen::Index nextLogicalOffset = 0;
			Eigen::Index nextStorageOffset = 0;
			if (!CheckedDimensionsDetail::checkedAdd(result.TotalDimension, dimension, nextLogicalOffset)
			    || !CheckedDimensionsDetail::checkedAdd(result.StoredSize, blockElements, nextStorageOffset))
				return CheckedDimensionsDetail::rejectInvalidDimensions();

			result.Dimensions.push_back(dimension);
			result.BlockOffsets.push_back(nextLogicalOffset);
			result.StorageOffsets.push_back(nextStorageOffset);
			result.TotalDimension = nextLogicalOffset;
			result.StoredSize = nextStorageOffset;
		}

		Eigen::Index representedSize = 0;
		if (StorageRule::CheckRepresentedSquare && result.TotalDimension != 0
		    && !CheckedDimensionsDetail::checkedSquare(result.TotalDimension, representedSize))
			return CheckedDimensionsDetail::rejectInvalidDimensions();
		return result;
	}

	template <typename Iterator,
	          std::enable_if_t<CheckedDimensionsDetail::is_supported_iterator<Iterator>::value, int> = 0>
	CheckedDimensions BuildCheckedDimensions(Iterator first, Iterator last)
	{
		return BuildCheckedDimensionsWith<CheckedDimensionsDetail::DenseStorageRule>(first, last);
	}

	template <typename Iterator,
	          std::enable_if_t<CheckedDimensionsDetail::is_supported_iterator<Iterator>::value, int> = 0>
	CheckedDimensions BuildCheckedVectorDimensions(Iterator first, Iterator last)
	{
		return BuildCheckedDimensionsWith<CheckedDimensionsDetail::VectorStorageRule>(first, last);
	}
}  // namespace Hoppy::Detail
