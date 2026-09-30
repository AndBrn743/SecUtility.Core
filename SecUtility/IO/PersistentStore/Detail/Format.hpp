// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Directory.hpp>
#include <SecUtility/Raw/Int.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <string_view>


namespace SecUtility::IO::PersistentStoreDetail
{
	inline constexpr UInt64 SuperblockBytes = 4096;
	inline constexpr UInt64 HeaderSlotBytes = 128;
	inline constexpr UInt64 HeaderSlotAOffset = 0;
	inline constexpr UInt64 HeaderSlotBOffset = 128;
	inline constexpr UInt64 DirectoryHeaderBytes = 40;
	inline constexpr UInt64 DirectoryEntryFixedBytes = 40;
	inline constexpr UInt64 MaximumKeyBytes = 1024;
	inline constexpr UInt64 MaximumPayloadAlignment = 4096;
	inline constexpr UInt64 MaximumFileBytes = (UInt64{1} << 63) - 1;
	inline constexpr UInt64 MaximumDirectoryBytes = 256 * 1024 * 1024;
	inline constexpr UInt64 MaximumDirectoryEntries = 1024 * 1024;

	inline constexpr UInt16 FormatMajor = 1;
	inline constexpr UInt16 FormatMinor = 0;
	inline constexpr UInt32 ByteOrderMarker = 0x01020304;
	inline constexpr UInt32 DirectoryFormatVersion = 1;

	inline constexpr std::array<Byte, 8> StoreMagic = {
	        Byte{'S'}, Byte{'E'}, Byte{'C'}, Byte{'P'}, Byte{'S'}, Byte{'R'}, Byte{'1'}, Byte{0}};
	inline constexpr std::array<Byte, 8> DirectoryMagic = {
	        Byte{'S'}, Byte{'E'}, Byte{'C'}, Byte{'D'}, Byte{'I'}, Byte{'R'}, Byte{'1'}, Byte{0}};

	static_assert(HeaderSlotBOffset + HeaderSlotBytes <= SuperblockBytes, "header slots must fit in the superblock");
	static_assert(DirectoryEntryFixedBytes == 40, "directory entry wire size changed");
	static_assert(DirectoryHeaderBytes == 40, "directory header wire size changed");

	struct HeaderSlot
	{
		UInt64 Generation = 0;
		UInt64 DirectoryOffset = 0;
		UInt64 DirectoryBytes = 0;
		UInt64 CommittedLogicalFileBytes = 0;
		UInt32 DirectoryChecksum = 0;
	};

	struct RootCandidate
	{
		bool IsValid = false;
		UInt64 SlotOffset = 0;
		HeaderSlot Header;
		Directory ParsedDirectory;
		std::string RejectionReason;
	};

	inline bool IsValidUtf8(const std::string_view text) noexcept
	{
		std::size_t index = 0;
		while (index < text.size())
		{
			const auto first = static_cast<unsigned char>(text[index]);
			if (first <= 0x7F)
			{
				++index;
				continue;
			}

			std::size_t continuationCount = 0;
			UInt32 codePoint = 0;
			UInt32 minimumCodePoint = 0;
			if (first >= 0xC2 && first <= 0xDF)
			{
				continuationCount = 1;
				codePoint = first & 0x1F;
				minimumCodePoint = 0x80;
			}
			else if (first >= 0xE0 && first <= 0xEF)
			{
				continuationCount = 2;
				codePoint = first & 0x0F;
				minimumCodePoint = 0x800;
			}
			else if (first >= 0xF0 && first <= 0xF4)
			{
				continuationCount = 3;
				codePoint = first & 0x07;
				minimumCodePoint = 0x10000;
			}
			else
			{
				return false;
			}

			if (continuationCount > text.size() - index - 1)
			{
				return false;
			}
			for (std::size_t continuation = 0; continuation < continuationCount; ++continuation)
			{
				const auto byte = static_cast<unsigned char>(text[index + continuation + 1]);
				if ((byte & 0xC0) != 0x80)
				{
					return false;
				}
				codePoint = (codePoint << 6) | (byte & 0x3F);
			}

			if (codePoint < minimumCodePoint || codePoint > 0x10FFFF || (codePoint >= 0xD800 && codePoint <= 0xDFFF))
			{
				return false;
			}
			index += continuationCount + 1;
		}
		return true;
	}

	inline void ValidateKey(const std::string_view key)
	{
		if (key.size() > MaximumKeyBytes)
		{
			throw InvalidKeyException("PersistentStore key exceeds 1024 UTF-8 bytes", key.size());
		}
		if (!IsValidUtf8(key))
		{
			throw InvalidKeyException("PersistentStore key is not valid UTF-8");
		}
	}

	inline void ValidateStoredKey(const std::string_view key)
	{
		if (key.size() > MaximumKeyBytes)
		{
			throw FormatException("stored key exceeds 1024 UTF-8 bytes", key.size());
		}
		if (!IsValidUtf8(key))
		{
			throw FormatException("stored key is not valid UTF-8");
		}
	}
}
