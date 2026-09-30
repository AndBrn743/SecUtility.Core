// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Raw/Int.hpp>

#include <limits>
#include <string_view>
#include <type_traits>


namespace SecUtility::IO::PersistentStoreDetail
{
	inline UInt64 CheckedAdd(const UInt64 left, const UInt64 right, const std::string_view context)
	{
		if (right > std::numeric_limits<UInt64>::max() - left)
		{
			throw FormatException(context, "unsigned 64-bit addition overflow");
		}
		return left + right;
	}

	inline UInt64 CheckedMultiply(const UInt64 left, const UInt64 right, const std::string_view context)
	{
		if (left != 0 && right > std::numeric_limits<UInt64>::max() / left)
		{
			throw FormatException(context, "unsigned 64-bit multiplication overflow");
		}
		return left * right;
	}

	inline UInt64 CheckedAlignUp(const UInt64 value, const UInt64 alignment, const std::string_view context)
	{
		if (!Math::IsPowerOfTwo(alignment))
		{
			throw FormatException(context, "alignment must be a nonzero power of two");
		}

		const UInt64 mask = alignment - 1;
		return CheckedAdd(value, mask, context) & ~mask;
	}

	template <typename T>
	T CheckedNarrow(const UInt64 value, const std::string_view context)
	{
		static_assert(std::is_integral_v<T>, "CheckedNarrow requires an integral destination");
		if constexpr (std::is_signed_v<T>)
		{
			if (value > static_cast<UInt64>(std::numeric_limits<T>::max()))
			{
				throw FormatException(context, "integer narrowing overflow");
			}
		}
		else if (value > static_cast<UInt64>(std::numeric_limits<T>::max()))
		{
			throw FormatException(context, "integer narrowing overflow");
		}
		return static_cast<T>(value);
	}

	inline bool IsRangeContained(const UInt64 outerOffset,
	                             const UInt64 outerLength,
	                             const UInt64 innerOffset,
	                             const UInt64 innerLength) noexcept
	{
		if (outerOffset > std::numeric_limits<UInt64>::max() - outerLength
		    || innerOffset > std::numeric_limits<UInt64>::max() - innerLength)
		{
			return false;
		}

		return innerOffset >= outerOffset && innerOffset + innerLength <= outerOffset + outerLength;
	}

	inline bool DoRangesOverlap(const UInt64 leftOffset,
	                            const UInt64 leftLength,
	                            const UInt64 rightOffset,
	                            const UInt64 rightLength) noexcept
	{
		if (leftLength == 0 || rightLength == 0 || leftOffset > std::numeric_limits<UInt64>::max() - leftLength
		    || rightOffset > std::numeric_limits<UInt64>::max() - rightLength)
		{
			return false;
		}
		return leftOffset < rightOffset + rightLength && rightOffset < leftOffset + leftLength;
	}
}
