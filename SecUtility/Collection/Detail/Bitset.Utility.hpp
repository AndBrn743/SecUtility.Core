// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if !defined(SEC_BITSET_DETAIL)
#error "Please do not directly include internal headers"
#endif

#include <cinttypes>
#include <cstddef>


namespace SecUtility::Detail::Bitset
{
	inline constexpr std::size_t BitsPerBlock = 64;

	constexpr std::size_t BlocksFor(const std::size_t bits) noexcept
	{
		return (bits + BitsPerBlock - 1) / BitsPerBlock;
	}

	// Mask of the active bits in the last block.  Returns ~0ull when bits % 64 == 0.
	constexpr std::uint64_t TailMask(const std::size_t bits) noexcept
	{
		const std::size_t remainder = bits % BitsPerBlock;
		return remainder == 0 ? ~std::uint64_t{0} : (std::uint64_t{1} << remainder) - 1u;
	}

	constexpr std::uint64_t HeadMask(const std::size_t paddingBitsCount) noexcept
	{
		return ~((std::uint64_t{1} << paddingBitsCount) - 1u);
	}

	constexpr std::size_t PopCount(std::uint64_t v) noexcept
	{
#if defined(__GNUC__) || defined(__clang__)
		// `long long` is at least 64 bits
		// ReSharper disable once CppRedundantCastExpression
		return static_cast<std::size_t>(__builtin_popcountll(static_cast<unsigned long long>(v)));
#elif defined(_MSC_VER)
		return static_cast<std::size_t>(__popcnt64(v));
#else
		v = v - ((v >> 1) & 0x5555555555555555ull);
		v = (v & 0x3333333333333333ull) + ((v >> 2) & 0x3333333333333333ull);
		v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0Full;
		return static_cast<std::size_t>((v * 0x0101010101010101ull) >> 56);
#endif
	}

	template <typename T>
	std::size_t PopCount(T) = delete;  // to prevent conversion

	constexpr std::size_t TrailingZeroCount(std::uint64_t v) noexcept
	{
		if (v == 0)
		{
			return 64;
		}
#if defined(__GNUC__) || defined(__clang__)
		// `long long` is at least 64 bits
		// ReSharper disable once CppRedundantCastExpression
		return static_cast<std::size_t>(__builtin_ctzll(static_cast<unsigned long long>(v)));
#elif defined(_MSC_VER)
		return static_cast<std::size_t>(_tzcnt_u64(v));
#else
		std::size_t n = 0;
		while ((v & 1u) == 0)
		{
			v >>= 1;
			++n;
		}
		return n;
#endif
	}

	template <typename T>
	std::size_t TrailingZeroCount(T) = delete;  // to prevent conversion

	constexpr std::size_t LeadingZeroCount(std::uint64_t v) noexcept
	{
		if (v == 0)
		{
			return 64;
		}
#if defined(__GNUC__) || defined(__clang__)
		// `long long` is at least 64 bits
		// ReSharper disable once CppRedundantCastExpression
		return static_cast<std::size_t>(__builtin_clzll(static_cast<unsigned long long>(v)));
#elif defined(_MSC_VER)
		return static_cast<std::size_t>(_lzcnt_u64(v));
#else
		std::size_t n = 0;
		while ((v & (std::uint64_t{1} << 63)) == 0)
		{
			v <<= 1;
			++n;
		}
		return n;
#endif
	}

	template <typename T>
	std::size_t LeadingZeroCount(T) = delete;  // to prevent conversion
}
