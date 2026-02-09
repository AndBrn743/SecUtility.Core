// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <cstdint>

using ulong = unsigned long;
using ushort = unsigned short;
using uint = unsigned int;

#if defined(SECUTILITY_NO_TOP_LEVEL_INT_ALIAS) && SECUTILITY_NO_TOP_LEVEL_INT_ALIAS
namespace SecUtility
{
#endif

	using Int8 = std::int8_t;
	using Int16 = std::int16_t;
	using Int32 = std::int32_t;
	using Int64 = std::int64_t;

	using UInt8 = std::uint8_t;
	using UInt16 = std::uint16_t;
	using UInt32 = std::uint32_t;
	using UInt64 = std::uint64_t;

	enum class Byte : unsigned char
	{
		/* NO CODE  */
	};

	enum class SByte : signed char
	{
		/* NO CODE  */
	};

#if defined(SECUTILITY_NO_TOP_LEVEL_INT_ALIAS) && SECUTILITY_NO_TOP_LEVEL_INT_ALIAS
}
#endif
