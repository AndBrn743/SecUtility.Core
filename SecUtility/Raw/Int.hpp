// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <cstdint>
#include <cstddef>

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


namespace SecUtility
{
	namespace Detail::Raw
	{
		template <std::size_t>
		struct integer_type_with_size;

		template <std::size_t S>
		using integer_type_with_size_t = typename integer_type_with_size<S>::type;

		template <std::size_t>
		struct unsigned_integer_type_with_size;

		template <std::size_t S>
		using unsigned_integer_type_with_size_t = typename unsigned_integer_type_with_size<S>::type;

		template <>
		struct integer_type_with_size<8>
		{
			using type = std::int8_t;
		};

		template <>
		struct integer_type_with_size<16>
		{
			using type = std::int16_t;
		};

		template <>
		struct integer_type_with_size<32>
		{
			using type = std::int32_t;
		};

		template <>
		struct integer_type_with_size<64>
		{
			using type = std::int64_t;
		};

		template <>
		struct unsigned_integer_type_with_size<8>
		{
			using type = std::uint8_t;
		};

		template <>
		struct unsigned_integer_type_with_size<16>
		{
			using type = std::uint16_t;
		};

		template <>
		struct unsigned_integer_type_with_size<32>
		{
			using type = std::uint32_t;
		};

		template <>
		struct unsigned_integer_type_with_size<64>
		{
			using type = std::uint64_t;
		};
	}

	template <std::size_t SizeInBits>
	using SignedIntegral = typename Detail::Raw::integer_type_with_size<SizeInBits>::type;

	template <std::size_t SizeInBits>
	using UnsignedIntegral = typename Detail::Raw::unsigned_integer_type_with_size<SizeInBits>::type;
}
