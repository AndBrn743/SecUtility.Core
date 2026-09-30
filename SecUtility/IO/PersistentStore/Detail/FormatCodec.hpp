// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/IO/PersistentStore/Detail/CheckedArithmetic.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Format.hpp>
#include <SecUtility/Misc/Checksum.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>


namespace SecUtility::IO::PersistentStoreDetail
{
	class ConstByteView
	{
	public:
		ConstByteView(const Byte* dataPtr, const std::size_t size) : m_DataPtr(dataPtr), m_Size(size)
		{
			if (dataPtr == nullptr && size != 0)
			{
				throw FormatException("nonempty byte view has a null data pointer");
			}
		}

		template <std::size_t Size>
		explicit ConstByteView(const std::array<Byte, Size>& bytes) noexcept : m_DataPtr(bytes.data()), m_Size(Size)
		{
			/* NO CODE */
		}

		explicit ConstByteView(const std::vector<Byte>& bytes) noexcept : m_DataPtr(bytes.data()), m_Size(bytes.size())
		{
			/* NO CODE */
		}

		const Byte* data() const noexcept
		{
			return m_DataPtr;
		}
		std::size_t size() const noexcept
		{
			return m_Size;
		}

	private:
		const Byte* m_DataPtr;
		std::size_t m_Size;
	};

	class MutableByteView
	{
	public:
		MutableByteView(Byte* dataPtr, const std::size_t size) : m_DataPtr(dataPtr), m_Size(size)
		{
			if (dataPtr == nullptr && size != 0)
			{
				throw FormatException("nonempty mutable byte view has a null data pointer");
			}
		}

		template <std::size_t Size>
		explicit MutableByteView(std::array<Byte, Size>& bytes) noexcept : m_DataPtr(bytes.data()), m_Size(Size)
		{
			/* NO CODE */
		}

		explicit MutableByteView(std::vector<Byte>& bytes) noexcept : m_DataPtr(bytes.data()), m_Size(bytes.size())
		{
			/* NO CODE */
		}

		Byte* data() const noexcept
		{
			return m_DataPtr;
		}
		std::size_t size() const noexcept
		{
			return m_Size;
		}

	private:
		Byte* m_DataPtr;
		std::size_t m_Size;
	};

	namespace FormatCodecDetail
	{
		inline void RequireRange(const std::size_t offset,
		                         const std::size_t byteCount,
		                         const std::size_t availableBytes,
		                         const std::string_view context)
		{
			if (offset > availableBytes || byteCount > availableBytes - offset)
			{
				throw FormatException(context, "truncated field");
			}
		}

		template <typename T>
		T ReadLittleEndian(const ConstByteView bytes, const std::size_t offset, const std::string_view context)
		{
			static_assert(std::is_unsigned_v<T>, "wire integers must decode through an unsigned type");
			RequireRange(offset, sizeof(T), bytes.size(), context);
			T value = 0;
			for (std::size_t index = 0; index < sizeof(T); ++index)
			{
				value |= static_cast<T>(std::to_integer<unsigned char>(bytes.data()[offset + index])) << (index * 8);
			}
			return value;
		}

		template <typename T>
		void WriteLittleEndian(const MutableByteView bytes,
		                       const std::size_t offset,
		                       const T value,
		                       const std::string_view context)
		{
			static_assert(std::is_unsigned_v<T>, "wire integers must encode through an unsigned type");
			RequireRange(offset, sizeof(T), bytes.size(), context);
			for (std::size_t index = 0; index < sizeof(T); ++index)
			{
				bytes.data()[offset + index] = Byte{static_cast<unsigned char>(value >> (index * 8))};
			}
		}

		inline UInt32 ComputeCrc32C(const ConstByteView bytes)
		{
			return static_cast<UInt32>(Checksum::Crc32C(bytes.data(), bytes.size()));
		}

		inline void RequireZero(const ConstByteView bytes,
		                        const std::size_t offset,
		                        const std::size_t byteCount,
		                        const std::string_view context)
		{
			RequireRange(offset, byteCount, bytes.size(), context);
			for (std::size_t index = 0; index < byteCount; ++index)
			{
				if (bytes.data()[offset + index] != Byte{0})
				{
					throw FormatException(context, "reserved bytes must be zero");
				}
			}
		}

		inline bool IsUnsignedByteLess(const std::string_view left, const std::string_view right) noexcept
		{
			return std::lexicographical_compare(
			        left.begin(),
			        left.end(),
			        right.begin(),
			        right.end(),
			        [](const char leftByte, const char rightByte)
			        { return static_cast<unsigned char>(leftByte) < static_cast<unsigned char>(rightByte); });
		}
	}

	inline std::array<Byte, HeaderSlotBytes> SerializeHeader(const HeaderSlot& header)
	{
		if (header.Generation == 0)
		{
			throw FormatException("cannot serialize header generation zero");
		}
		if (header.CommittedLogicalFileBytes < SuperblockBytes || header.CommittedLogicalFileBytes > MaximumFileBytes
		    || header.DirectoryOffset < SuperblockBytes || header.DirectoryBytes < DirectoryHeaderBytes
		    || header.DirectoryBytes > MaximumDirectoryBytes
		    || !IsRangeContained(0, header.CommittedLogicalFileBytes, header.DirectoryOffset, header.DirectoryBytes))
		{
			throw FormatException("cannot serialize an invalid directory root");
		}

		std::array<Byte, HeaderSlotBytes> bytes{};
		std::copy(StoreMagic.begin(), StoreMagic.end(), bytes.begin());
		const MutableByteView view(bytes);
		FormatCodecDetail::WriteLittleEndian<UInt16>(view, 8, FormatMajor, "format major");
		FormatCodecDetail::WriteLittleEndian<UInt16>(view, 10, FormatMinor, "format minor");
		FormatCodecDetail::WriteLittleEndian<UInt32>(view, 12, ByteOrderMarker, "byte-order marker");
		FormatCodecDetail::WriteLittleEndian<UInt32>(view, 16, HeaderSlotBytes, "header bytes");
		FormatCodecDetail::WriteLittleEndian<UInt32>(view, 20, 0, "required feature flags");
		FormatCodecDetail::WriteLittleEndian<UInt64>(view, 24, header.Generation, "generation");
		FormatCodecDetail::WriteLittleEndian<UInt64>(view, 32, header.DirectoryOffset, "directory offset");
		FormatCodecDetail::WriteLittleEndian<UInt64>(view, 40, header.DirectoryBytes, "directory bytes");
		FormatCodecDetail::WriteLittleEndian<UInt64>(
		        view, 48, header.CommittedLogicalFileBytes, "committed logical file bytes");
		FormatCodecDetail::WriteLittleEndian<UInt32>(view, 56, header.DirectoryChecksum, "directory checksum");
		const UInt32 checksum = FormatCodecDetail::ComputeCrc32C(ConstByteView(bytes));
		FormatCodecDetail::WriteLittleEndian<UInt32>(view, 60, checksum, "header checksum");
		return bytes;
	}

	inline HeaderSlot ParseHeader(const ConstByteView bytes, const UInt64 physicalFileBytes)
	{
		if (bytes.size() != HeaderSlotBytes)
		{
			throw FormatException("header slot must contain exactly 128 bytes", bytes.size());
		}
		if (!std::equal(StoreMagic.begin(), StoreMagic.end(), bytes.data()))
		{
			throw FormatException("unsupported or corrupt PersistentStore magic");
		}

		const auto formatMajor = FormatCodecDetail::ReadLittleEndian<UInt16>(bytes, 8, "format major");
		const auto formatMinor = FormatCodecDetail::ReadLittleEndian<UInt16>(bytes, 10, "format minor");
		if (formatMajor != FormatMajor || formatMinor != FormatMinor)
		{
			throw FormatException("unsupported PersistentStore format version", formatMajor, formatMinor);
		}
		if (FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 12, "byte-order marker") != ByteOrderMarker)
		{
			throw FormatException("invalid PersistentStore byte-order marker");
		}
		if (FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 16, "header bytes") != HeaderSlotBytes)
		{
			throw FormatException("unsupported PersistentStore header size");
		}
		if (FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 20, "required feature flags") != 0)
		{
			throw FormatException("unsupported PersistentStore required feature flags");
		}
		FormatCodecDetail::RequireZero(bytes, 64, 64, "header reserved bytes");

		std::array<Byte, HeaderSlotBytes> checksumBytes{};
		std::copy(bytes.data(), bytes.data() + bytes.size(), checksumBytes.begin());
		const UInt32 storedHeaderChecksum = FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 60, "header checksum");
		std::fill(checksumBytes.begin() + 60, checksumBytes.begin() + 64, Byte{0});
		if (FormatCodecDetail::ComputeCrc32C(ConstByteView(checksumBytes)) != storedHeaderChecksum)
		{
			throw FormatException("PersistentStore header checksum mismatch");
		}

		HeaderSlot header;
		header.Generation = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 24, "generation");
		header.DirectoryOffset = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 32, "directory offset");
		header.DirectoryBytes = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 40, "directory bytes");
		header.CommittedLogicalFileBytes =
		        FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 48, "committed logical file bytes");
		header.DirectoryChecksum = FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 56, "directory checksum");

		if (header.Generation == 0)
		{
			throw FormatException("PersistentStore header generation zero is invalid");
		}
		if (physicalFileBytes > MaximumFileBytes || header.CommittedLogicalFileBytes > physicalFileBytes
		    || header.CommittedLogicalFileBytes < SuperblockBytes)
		{
			throw FormatException("PersistentStore committed length is outside the physical file");
		}
		if (header.DirectoryBytes < DirectoryHeaderBytes || header.DirectoryBytes > MaximumDirectoryBytes
		    || !IsRangeContained(0, header.CommittedLogicalFileBytes, header.DirectoryOffset, header.DirectoryBytes)
		    || header.DirectoryOffset < SuperblockBytes)
		{
			throw FormatException("PersistentStore directory range is invalid");
		}
		return header;
	}

	inline std::vector<Byte> SerializeDirectory(const Directory& directory)
	{
		if (directory.Generation == 0 || directory.Entries.size() > MaximumDirectoryEntries)
		{
			throw FormatException("directory generation or entry count is invalid");
		}

		UInt64 totalBytes = DirectoryHeaderBytes;
		const DirectoryEntry* previousEntryPtr = nullptr;
		std::vector<Extent> allocatedExtents;
		allocatedExtents.reserve(directory.Entries.size());
		for (const DirectoryEntry& entry : directory.Entries)
		{
			ValidateStoredKey(entry.Key);
			if (entry.AllocatedExtent.Capacity == 0 || entry.AllocatedExtent.Offset < SuperblockBytes
			    || !IsRangeContained(entry.AllocatedExtent.Offset,
			                         entry.AllocatedExtent.Capacity,
			                         entry.PayloadOffset,
			                         entry.PayloadBytes))
			{
				throw FormatException("cannot serialize an invalid entry extent", entry.Key);
			}
			if (previousEntryPtr != nullptr && !FormatCodecDetail::IsUnsignedByteLess(previousEntryPtr->Key, entry.Key))
			{
				throw FormatException("directory keys must be strictly sorted and unique");
			}
			previousEntryPtr = &entry;
			allocatedExtents.push_back(entry.AllocatedExtent);
			totalBytes = CheckedAdd(totalBytes, DirectoryEntryFixedBytes, "directory encoded length");
			totalBytes = CheckedAdd(totalBytes, entry.Key.size(), "directory encoded length");
		}
		if (totalBytes > MaximumDirectoryBytes)
		{
			throw FormatException("directory exceeds the encoded-length limit", totalBytes);
		}
		std::sort(allocatedExtents.begin(),
		          allocatedExtents.end(),
		          [](const Extent& left, const Extent& right) { return left.Offset < right.Offset; });
		for (std::size_t index = 1; index < allocatedExtents.size(); ++index)
		{
			const Extent& previous = allocatedExtents[index - 1];
			const Extent& current = allocatedExtents[index];
			if (DoRangesOverlap(previous.Offset, previous.Capacity, current.Offset, current.Capacity))
			{
				throw FormatException("cannot serialize overlapping payload allocations");
			}
		}

		std::vector<Byte> bytes(CheckedNarrow<std::size_t>(totalBytes, "directory vector size"), Byte{0});
		const MutableByteView view(bytes);
		std::copy(DirectoryMagic.begin(), DirectoryMagic.end(), bytes.begin());
		FormatCodecDetail::WriteLittleEndian<UInt32>(view, 8, DirectoryFormatVersion, "directory version");
		FormatCodecDetail::WriteLittleEndian<UInt32>(view, 12, DirectoryHeaderBytes, "directory header bytes");
		FormatCodecDetail::WriteLittleEndian<UInt64>(view, 16, directory.Generation, "directory generation");
		FormatCodecDetail::WriteLittleEndian<UInt64>(
		        view, 24, static_cast<UInt64>(directory.Entries.size()), "directory entry count");
		FormatCodecDetail::WriteLittleEndian<UInt64>(view, 32, totalBytes, "directory total bytes");

		std::size_t offset = DirectoryHeaderBytes;
		for (const DirectoryEntry& entry : directory.Entries)
		{
			FormatCodecDetail::WriteLittleEndian<UInt32>(
			        view, offset, static_cast<UInt32>(entry.Key.size()), "key bytes");
			FormatCodecDetail::WriteLittleEndian<UInt32>(view, offset + 4, 0, "entry reserved bytes");
			FormatCodecDetail::WriteLittleEndian<UInt64>(
			        view, offset + 8, entry.AllocatedExtent.Offset, "allocated extent offset");
			FormatCodecDetail::WriteLittleEndian<UInt64>(
			        view, offset + 16, entry.AllocatedExtent.Capacity, "allocated extent capacity");
			FormatCodecDetail::WriteLittleEndian<UInt64>(view, offset + 24, entry.PayloadOffset, "payload offset");
			FormatCodecDetail::WriteLittleEndian<UInt64>(view, offset + 32, entry.PayloadBytes, "payload bytes");
			offset += DirectoryEntryFixedBytes;
			if (!entry.Key.empty())
			{
				std::memcpy(bytes.data() + offset, entry.Key.data(), entry.Key.size());
			}
			offset += entry.Key.size();
		}
		return bytes;
	}

	inline Directory ParseDirectory(const HeaderSlot& header, const ConstByteView bytes, const UInt64 physicalFileBytes)
	{
		if (physicalFileBytes > MaximumFileBytes || header.CommittedLogicalFileBytes > physicalFileBytes
		    || header.CommittedLogicalFileBytes < SuperblockBytes || header.DirectoryOffset < SuperblockBytes
		    || !IsRangeContained(0, header.CommittedLogicalFileBytes, header.DirectoryOffset, header.DirectoryBytes))
		{
			throw FormatException("directory root is outside the committed or physical file");
		}
		if (bytes.size() != header.DirectoryBytes || bytes.size() < DirectoryHeaderBytes
		    || bytes.size() > MaximumDirectoryBytes)
		{
			throw FormatException("directory byte length does not match its root");
		}
		if (FormatCodecDetail::ComputeCrc32C(bytes) != header.DirectoryChecksum)
		{
			throw FormatException("directory checksum mismatch");
		}
		if (!std::equal(DirectoryMagic.begin(), DirectoryMagic.end(), bytes.data()))
		{
			throw FormatException("unsupported or corrupt directory magic");
		}
		if (FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 8, "directory version") != DirectoryFormatVersion)
		{
			throw FormatException("unsupported directory format version");
		}
		if (FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, 12, "directory header bytes") != DirectoryHeaderBytes)
		{
			throw FormatException("unsupported directory header size");
		}

		Directory directory;
		directory.Generation = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 16, "directory generation");
		if (directory.Generation != header.Generation)
		{
			throw FormatException("directory generation does not match its root");
		}
		const UInt64 entryCount = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 24, "directory entry count");
		const UInt64 encodedBytes = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, 32, "directory total bytes");
		if (encodedBytes != bytes.size() || entryCount > MaximumDirectoryEntries
		    || entryCount > (encodedBytes - DirectoryHeaderBytes) / DirectoryEntryFixedBytes)
		{
			throw FormatException("directory count or encoded length is invalid");
		}

		directory.Entries.reserve(CheckedNarrow<std::size_t>(entryCount, "directory entry count"));
		std::size_t offset = DirectoryHeaderBytes;
		for (UInt64 index = 0; index < entryCount; ++index)
		{
			FormatCodecDetail::RequireRange(offset, DirectoryEntryFixedBytes, bytes.size(), "directory entry");
			const UInt32 keyBytes = FormatCodecDetail::ReadLittleEndian<UInt32>(bytes, offset, "key bytes");
			if (keyBytes > MaximumKeyBytes)
			{
				throw FormatException("stored key exceeds 1024 UTF-8 bytes", keyBytes);
			}
			FormatCodecDetail::RequireZero(bytes, offset + 4, 4, "directory entry reserved bytes");

			DirectoryEntry entry;
			entry.AllocatedExtent.Offset =
			        FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, offset + 8, "allocated extent offset");
			entry.AllocatedExtent.Capacity =
			        FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, offset + 16, "allocated extent capacity");
			entry.PayloadOffset = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, offset + 24, "payload offset");
			entry.PayloadBytes = FormatCodecDetail::ReadLittleEndian<UInt64>(bytes, offset + 32, "payload bytes");
			offset += DirectoryEntryFixedBytes;
			FormatCodecDetail::RequireRange(offset, keyBytes, bytes.size(), "directory key");
			entry.Key.assign(reinterpret_cast<const char*>(bytes.data() + offset), keyBytes);
			ValidateStoredKey(entry.Key);
			offset += keyBytes;
			directory.Entries.push_back(std::move(entry));
		}
		if (offset != bytes.size())
		{
			throw FormatException("directory contains trailing or unconsumed bytes");
		}

		const DirectoryEntry* previousEntryPtr = nullptr;
		for (const DirectoryEntry& entry : directory.Entries)
		{
			if (previousEntryPtr != nullptr && !FormatCodecDetail::IsUnsignedByteLess(previousEntryPtr->Key, entry.Key))
			{
				throw FormatException("directory keys are duplicated or out of order");
			}
			previousEntryPtr = &entry;
			if (entry.AllocatedExtent.Capacity == 0 || entry.AllocatedExtent.Offset < SuperblockBytes
			    || !IsRangeContained(0,
			                         header.CommittedLogicalFileBytes,
			                         entry.AllocatedExtent.Offset,
			                         entry.AllocatedExtent.Capacity)
			    || !IsRangeContained(entry.AllocatedExtent.Offset,
			                         entry.AllocatedExtent.Capacity,
			                         entry.PayloadOffset,
			                         entry.PayloadBytes))
			{
				throw FormatException("directory entry extent or payload range is invalid", entry.Key);
			}
		}

		if (physicalFileBytes < header.CommittedLogicalFileBytes)
		{
			throw FormatException("directory root exceeds physical file length");
		}
		std::vector<Extent> allocatedExtents;
		allocatedExtents.reserve(directory.Entries.size());
		for (const DirectoryEntry& entry : directory.Entries)
		{
			if (DoRangesOverlap(header.DirectoryOffset,
			                    header.DirectoryBytes,
			                    entry.AllocatedExtent.Offset,
			                    entry.AllocatedExtent.Capacity))
			{
				throw FormatException("directory metadata overlaps a payload allocation");
			}
			allocatedExtents.push_back(entry.AllocatedExtent);
		}
		std::sort(allocatedExtents.begin(),
		          allocatedExtents.end(),
		          [](const Extent& left, const Extent& right) { return left.Offset < right.Offset; });
		for (std::size_t index = 1; index < allocatedExtents.size(); ++index)
		{
			const Extent& previous = allocatedExtents[index - 1];
			const Extent& current = allocatedExtents[index];
			if (DoRangesOverlap(previous.Offset, previous.Capacity, current.Offset, current.Capacity))
			{
				throw FormatException("directory contains overlapping payload allocations");
			}
		}
		return directory;
	}

	inline RootCandidate ParseRootCandidate(const UInt64 slotOffset,
	                                        const ConstByteView headerBytes,
	                                        const ConstByteView directoryBytes,
	                                        const UInt64 physicalFileBytes)
	{
		RootCandidate candidate;
		candidate.SlotOffset = slotOffset;
		try
		{
			candidate.Header = ParseHeader(headerBytes, physicalFileBytes);
			candidate.ParsedDirectory = ParseDirectory(candidate.Header, directoryBytes, physicalFileBytes);
			candidate.IsValid = true;
		}
		catch (const FormatException& exception)
		{
			candidate.RejectionReason = exception.what();
		}
		return candidate;
	}
}
