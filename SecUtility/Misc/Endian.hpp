// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <ostream>


namespace SecUtility
{
	enum class Endian
	{
#if defined(_MSC_VER) && !defined(__clang__)
		Little,
		Middle,
		Big,
		Pdp = Middle,
		Native = Little
#else
		Little = __ORDER_LITTLE_ENDIAN__,
		Middle = __ORDER_PDP_ENDIAN__,
		Big = __ORDER_BIG_ENDIAN__,
		Pdp = Middle,
		Native = __BYTE_ORDER__
#endif
	};

	inline std::ostream& operator<<(std::ostream& os, const Endian endian)
	{
		switch (endian)
		{
			case Endian::Little:
				return os << "Little";
			case Endian::Middle:
				return os << "Middle";
			case Endian::Big:
				return os << "Big";
			default:
				return os << "Unknown";
		}
	}
}
