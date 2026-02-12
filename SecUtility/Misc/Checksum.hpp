// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Misc/Enum.hpp>
#include <SecUtility/Raw/Int.hpp>
#include <iomanip>
#include <ostream>


namespace SecUtility::Checksum
{
	enum class Checksum32 : UInt32
	{
		/* NO CODE */
	};

	enum class Checksum64 : UInt64
	{
		/* NO CODE */
	};

	template <typename T>
	constexpr auto operator^(const T lhs, const Checksum32 rhs) noexcept
	{
		return lhs ^ std::to_underlying(rhs);
	}

	template <typename T>
	constexpr auto operator^(const Checksum32 lhs, const T rhs) noexcept
	{
		return std::to_underlying(lhs) ^ rhs;
	}

	template <typename T>
	constexpr auto operator>>(const Checksum32 lhs, const T rhs) noexcept
	{
		return std::to_underlying(lhs) >> rhs;
	}

	template <typename T>
	constexpr auto operator<<(const Checksum32 lhs, const T rhs) noexcept
	{
		return std::to_underlying(lhs) << rhs;
	}

	constexpr auto operator~(const Checksum32 checksum) noexcept
	{
		return ~std::to_underlying(checksum);
	}

	inline std::ostream& operator<<(std::ostream& os, const Checksum32 checksum)
	{
		const std::ios_base::fmtflags formatFlags = os.flags();
		const char previousFillChar = os.fill();

		os << "0x" << std::hex << std::noshowbase << std::uppercase << std::setfill('0') << std::setw(8)
		   << std::to_underlying(checksum);

		os.flags(formatFlags);
		os.fill(previousFillChar);

		return os;
	}

	template <typename T>
	constexpr auto operator^(const T lhs, const Checksum64 rhs) noexcept
	{
		return lhs ^ std::to_underlying(rhs);
	}

	template <typename T>
	constexpr auto operator^(const Checksum64 lhs, const T rhs) noexcept
	{
		return std::to_underlying(lhs) ^ rhs;
	}

	template <typename T>
	constexpr auto operator>>(const Checksum64 lhs, const T rhs) noexcept
	{
		return std::to_underlying(lhs) >> rhs;
	}

	template <typename T>
	constexpr auto operator<<(const Checksum64 lhs, const T rhs) noexcept
	{
		return std::to_underlying(lhs) << rhs;
	}

	constexpr auto operator~(const Checksum64 checksum) noexcept
	{
		return ~std::to_underlying(checksum);
	}

	inline std::ostream& operator<<(std::ostream& os, const Checksum64 checksum)
	{
		const std::ios_base::fmtflags formatFlags = os.flags();
		const char previousFillChar = os.fill();

		os << "0x" << std::hex << std::noshowbase << std::uppercase << std::setfill('0') << std::setw(16)
		   << std::to_underlying(checksum);

		os.flags(formatFlags);
		os.fill(previousFillChar);

		return os;
	}
}
