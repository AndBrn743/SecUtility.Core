// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Macro/ConstevalIf.hpp>
#include <SecUtility/Misc/Endian.hpp>
#include <SecUtility/Misc/Enum.hpp>
#include <SecUtility/Misc/Prefetch.hpp>
#include <SecUtility/Raw/Int.hpp>

#include <array>
#include <iomanip>
#include <ostream>

#if defined(__SSE4_2__)
#include <nmmintrin.h>  // CRC-32C intrinsics (GCC/Clang on x86/x64 with -msse4.2)
#define SECUTILITY_HAS_X86_CRC32C true
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>  // CRC-32C intrinsics (MSVC/clang-cl on Windows x86/x64)
#define SECUTILITY_HAS_X86_CRC32C true
#elif defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>  // CRC-32C intrinsics (ARMv8 + CRC32 extension)
#define SECUTILITY_HAS_ARM_CRC32C true
#endif

#if (defined(SECUTILITY_HAS_X86_CRC32C) && SECUTILITY_HAS_X86_CRC32C)                                                  \
        || (defined(SECUTILITY_HAS_ARM_CRC32C) && SECUTILITY_HAS_ARM_CRC32C)
#define SECUTILITY_HAS_HARDWARE_CRC32C true
#endif

// clang-cl mimics MSVC's /arch model and doesn't enable SSE4.2 globally even when
// the intrinsics are callable. Function-level target attribution is required for
// clang to emit the SSE4.2 instructions; MSVC doesn't check features, and POSIX
// clang gets the feature globally via -msse4.2 (so the attribute is redundant).
#if defined(__clang__) && defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)) && !defined(__SSE4_2__)
#define SECUTILITY_CRC32C_X86_TARGET __attribute__((target("sse4.2")))
#else
#define SECUTILITY_CRC32C_X86_TARGET
#endif

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
		const std::ios::fmtflags formatFlags = os.flags();
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
		const std::ios::fmtflags formatFlags = os.flags();
		const char previousFillChar = os.fill();

		os << "0x" << std::hex << std::noshowbase << std::uppercase << std::setfill('0') << std::setw(16)
		   << std::to_underlying(checksum);

		os.flags(formatFlags);
		os.fill(previousFillChar);

		return os;
	}
}


namespace SecUtility::Checksum
{
	// 0xEDB88320 for IEEE CRC-32, 0x82F63B78 for CRC-32C (Castagnoli)
	template <UInt32 Crc32Polynomial>
	inline constexpr auto Crc32Table = []
	{
		const auto GenerateCrcEntry = [](const UInt32 i)
		{
			UInt32 crc = i;
			for (int j = 0; j < 8; ++j)
			{
				crc = (crc >> 1) ^ ((crc & 1) ? Crc32Polynomial : 0);
			}
			return crc;
		};

		std::array<UInt32, 256> table = {};
		for (UInt32 i = 0; i < 256; ++i)
		{
			table[i] = GenerateCrcEntry(i);
		}
		return table;
	}();


	// 0xEDB88320 for IEEE CRC-32, 0x82F63B78 for CRC-32C (Castagnoli)
	template <UInt32 Crc32Polynomial, std::size_t SliceCount>
	inline constexpr auto SlicedCrc32Tables = []
	{
		std::array<std::array<UInt32, 256>, SliceCount> tables = {};

		tables[0] = Crc32Table<Crc32Polynomial>;

		for (std::size_t t = 1; t < SliceCount; ++t)
		{
			for (UInt32 i = 0; i < 256; ++i)
			{
				const UInt32 prev = tables[t - 1][i];
				tables[t][i] = (prev >> 8) ^ tables[0][prev & 0xFF];
			}
		}

		return tables;
	}();


	// 0xEDB88320 for IEEE CRC-32, 0x82F63B78 for CRC-32C (Castagnoli)
	template <UInt32 Crc32Polynomial, std::size_t SliceIndex>
	inline constexpr auto SlicedCrc32TableSlice = []
	{
		if constexpr (SliceIndex == 0)
		{
			return Crc32Table<Crc32Polynomial>;
		}
		else
		{
			std::array<UInt32, 256> table = {};

			for (UInt32 i = 0; i < 256; ++i)
			{
				const UInt32 prev = SlicedCrc32TableSlice<Crc32Polynomial, SliceIndex - 1>[i];
				table[i] = (prev >> 8) ^ Crc32Table<Crc32Polynomial>[prev & 0xFF];
			}

			return table;
		}
	}();

	// 0xEDB88320 for IEEE CRC-32, 0x82F63B78 for CRC-32C (Castagnoli)
	template <UInt32 Crc32Polynomial = 0xEDB88320>
	constexpr Checksum32 SoftwareCrc32(const UInt8* data,
	                                   const std::size_t byteCount,
	                                   Checksum32 crc = Checksum32{0xFFFFFFFF}) noexcept
	{
		for (std::size_t i = 0; i < byteCount; i++)
		{
			crc = Checksum32{Crc32Table<Crc32Polynomial>[(crc ^ data[i]) & 0xFF] ^ (crc >> 8)};
		}

		return Checksum32{crc ^ 0xFFFFFFFF};
	}

	constexpr Checksum32 SoftwareCrc32C(const UInt8* data,
	                                    const std::size_t byteCount,
	                                    const Checksum32 crc = Checksum32{0xFFFFFFFF}) noexcept
	{
		return SoftwareCrc32<0x82F63B78>(data, byteCount, crc);
	}

	// The slicing logic of this impl is adopted from https://create.stephan-brumme.com/crc32/ and
	// https://github.com/stbrumme/crc32 (Zlib license)
	template <UInt32 Crc32Polynomial = 0xEDB88320, std::size_t DegreeOfUnroll = 1>
	constexpr Checksum32 SlicedSoftwareCrc32(std::integral_constant<std::size_t, 8>,
	                                         const UInt8* data,
	                                         std::size_t byteCount,
	                                         const Checksum32 crc = Checksum32{0xFFFFFFFF})
	{
		auto _crc = static_cast<UInt32>(crc);

		while (byteCount >= 8 * DegreeOfUnroll)
		{
#if defined(SEC_IF_NOT_CONSTEVAL)
			SEC_IF_NOT_CONSTEVAL
#else
			if constexpr (false)
#endif
			{
				Prefetch(data + 8 * DegreeOfUnroll * 2);
			}

			for (std::size_t i = 0; i < DegreeOfUnroll; ++i)
			{
				const UInt32 block = static_cast<UInt32>(data[0]) | (static_cast<UInt32>(data[1]) << 8)
				                     | (static_cast<UInt32>(data[2]) << 16) | (static_cast<UInt32>(data[3]) << 24);

				const UInt32 one = block ^ _crc;

				_crc = SlicedCrc32Tables<Crc32Polynomial, 8>[0][data[7]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 8>[1][data[6]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 8>[2][data[5]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 8>[3][data[4]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 8>[4][(one >> 24) & 0xFF]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 8>[5][(one >> 16) & 0xFF]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 8>[6][(one >> 8) & 0xFF]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 8>[7][one & 0xFF];

				data += 8;
			}

			byteCount -= 8 * DegreeOfUnroll;
		}

		while (byteCount != 0)
		{
			_crc = Crc32Table<Crc32Polynomial>[(_crc ^ *data) & 0xFF] ^ (_crc >> 8);
			++data;
			--byteCount;
		}

		return static_cast<Checksum32>(_crc ^ 0xFFFFFFFF);
	}


	// The slicing logic of this impl is adopted from https://create.stephan-brumme.com/crc32/ and
	// https://github.com/stbrumme/crc32 (Zlib license)
	template <UInt32 Crc32Polynomial = 0xEDB88320, std::size_t DegreeOfUnroll = 1>
	constexpr Checksum32 SlicedSoftwareCrc32(std::integral_constant<std::size_t, 16>,
	                                         const UInt8* data,
	                                         std::size_t byteCount,
	                                         const Checksum32 crc = Checksum32{0xFFFFFFFF})
	{
		auto _crc = static_cast<UInt32>(crc);

		while (byteCount >= 16 * DegreeOfUnroll)
		{
#if defined(SEC_IF_NOT_CONSTEVAL)
			SEC_IF_NOT_CONSTEVAL
#else
			if constexpr (false)
#endif
			{
				Prefetch(data + 16 * DegreeOfUnroll * 2);
			}

			for (std::size_t i = 0; i < DegreeOfUnroll; ++i)
			{
				const UInt32 block = static_cast<UInt32>(data[0]) | (static_cast<UInt32>(data[1]) << 8)
				                     | (static_cast<UInt32>(data[2]) << 16) | (static_cast<UInt32>(data[3]) << 24);

				const UInt32 one = block ^ _crc;

				_crc = SlicedCrc32Tables<Crc32Polynomial, 16>[0][data[15]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[1][data[14]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[2][data[13]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[3][data[12]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[4][data[11]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[5][data[10]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[6][data[9]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[7][data[8]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[8][data[7]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[9][data[6]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[10][data[5]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[11][data[4]]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[12][(one >> 24) & 0xFF]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[13][(one >> 16) & 0xFF]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[14][(one >> 8) & 0xFF]
				       ^ SlicedCrc32Tables<Crc32Polynomial, 16>[15][one & 0xFF];
				data += 16;
			}

			byteCount -= 16 * DegreeOfUnroll;
		}

		while (byteCount != 0)
		{
			_crc = Crc32Table<Crc32Polynomial>[(_crc ^ *data) & 0xFF] ^ (_crc >> 8);
			++data;
			--byteCount;
		}

		return static_cast<Checksum32>(_crc ^ 0xFFFFFFFF);
	}


	template <std::size_t SliceCount, UInt32 Crc32Polynomial = 0xEDB88320>
	constexpr Checksum32 SlicedSoftwareCrc32(const uint8_t* data,
	                                         size_t byteCount,
	                                         const Checksum32 crc = Checksum32{0xFFFFFFFF})
	{
		return SlicedSoftwareCrc32<Crc32Polynomial>(
		        std::integral_constant<std::size_t, SliceCount>{}, data, byteCount, crc);
	}

	template <std::size_t SliceCount>
	constexpr Checksum32 SlicedSoftwareCrc32C(const uint8_t* data,
	                                          size_t byteCount,
	                                          const Checksum32 crc = Checksum32{0xFFFFFFFF})
	{
		return SlicedSoftwareCrc32<0x82F63B78>(std::integral_constant<std::size_t, SliceCount>{}, data, byteCount, crc);
	}


#if defined(SECUTILITY_HAS_HARDWARE_CRC32C) && SECUTILITY_HAS_HARDWARE_CRC32C
	SECUTILITY_CRC32C_X86_TARGET
	inline Checksum32 HardwareCrc32C(const UInt8* const data,
	                                 std::size_t byteCount,
	                                 const Checksum32 crc = Checksum32{0xFFFFFFFF}) noexcept
	{
		auto p = data;
		auto _crc = static_cast<UInt64>(crc);

		while (byteCount >= sizeof(UInt64))
		{
#if defined(SECUTILITY_HAS_X86_CRC32C) && SECUTILITY_HAS_X86_CRC32C
			_crc = _mm_crc32_u64(_crc, *reinterpret_cast<const UInt64*>(p));
#elif defined(SECUTILITY_HAS_ARM_CRC32C) && SECUTILITY_HAS_ARM_CRC32C
			_crc = __crc32cd(_crc, *reinterpret_cast<const UInt64*>(p));
#endif
			p += sizeof(UInt64);
			byteCount -= sizeof(UInt64);
		}

		while (byteCount--)
		{
#if defined(SECUTILITY_HAS_X86_CRC32C) && SECUTILITY_HAS_X86_CRC32C
			_crc = _mm_crc32_u8(static_cast<UInt32>(_crc), *p++);
#elif defined(SECUTILITY_HAS_ARM_CRC32C) && SECUTILITY_HAS_ARM_CRC32C
			_crc = __crc32cb(static_cast<UInt32>(_crc), *p++);
#endif
		}

		return static_cast<Checksum32>(_crc ^ 0xFFFFFFFF);
	}
#endif

	inline Checksum32 Crc32(const UInt8* data,
	                        const std::size_t byteCount,
	                        const Checksum32 crc = Checksum32{0xFFFFFFFF})
	{
		return SoftwareCrc32(data, byteCount, crc);
	}

	inline Checksum32 Crc32C(const UInt8* data,
	                         const std::size_t byteCount,
	                         const Checksum32 crc = Checksum32{0xFFFFFFFF})
	{
#if defined(SECUTILITY_HAS_HARDWARE_CRC32C) && SECUTILITY_HAS_HARDWARE_CRC32C
		if (byteCount >= 128)
		{
			return HardwareCrc32C(data, byteCount, crc);
		}
#endif

		return SlicedSoftwareCrc32<16, 4>(data, byteCount, crc);
	}
}
