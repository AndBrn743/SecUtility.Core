// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FileBackend.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FormatCodec.hpp>
#include <SecUtility/Misc/Endian.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>


namespace SecUtility::IO::PersistentStoreDetail
{
	class StoreState final
	{
	public:
		static std::shared_ptr<StoreState> Open(const std::filesystem::path& path, const FileAccess access)
		{
			RequireLittleEndianHost();
			return Load(FileBackend::Open(path, access));
		}

		static std::shared_ptr<StoreState> Create(const std::filesystem::path& path)
		{
			RequireLittleEndianHost();
			auto backend = FileBackend::TryCreateExclusive(path);
			if (!backend.has_value())
			{
				throw IOException("PersistentStore creation refused to overwrite an existing path");
			}
			return InitializeCreated(path, std::move(*backend));
		}

		static std::shared_ptr<StoreState> CreateIfNotExist(const std::filesystem::path& path)
		{
			RequireLittleEndianHost();
			auto backend = FileBackend::TryCreateExclusive(path);
			if (!backend.has_value())
			{
				return Load(FileBackend::Open(path, FileAccess::ReadWrite));
			}
			return InitializeCreated(path, std::move(*backend));
		}

		bool Contains(const std::string_view key) const
		{
			ValidateKey(key);
			const auto iterator = std::lower_bound(
			        m_CurrentDirectoryPtr->Entries.begin(), m_CurrentDirectoryPtr->Entries.end(), key,
			        [](const DirectoryEntry& entry, const std::string_view soughtKey) {
				        return FormatCodecDetail::IsUnsignedByteLess(entry.Key, soughtKey);
			        });
			return iterator != m_CurrentDirectoryPtr->Entries.end() && iterator->Key.size() == key.size()
			       && std::equal(iterator->Key.begin(), iterator->Key.end(), key.begin());
		}

		std::size_t Size() const noexcept { return m_CurrentDirectoryPtr->Entries.size(); }
		FileAccess GetAccess() const noexcept { return m_Backend.GetAccess(); }

		const RootCandidate& GetSlotA() const noexcept { return m_SlotA; }
		const RootCandidate& GetSlotB() const noexcept { return m_SlotB; }

	private:
		struct LoadedCandidate
		{
			RootCandidate Candidate;
			std::vector<Byte> DirectoryBytes;
		};

		StoreState(FileBackend backend,
		           RootCandidate slotA,
		           RootCandidate slotB,
		           std::shared_ptr<const Directory> currentDirectoryPtr)
		    : m_Backend(std::move(backend)),
		      m_SlotA(std::move(slotA)),
		      m_SlotB(std::move(slotB)),
		      m_CurrentDirectoryPtr(std::move(currentDirectoryPtr))
		{
			/* NO CODE */
		}

		static void RequireLittleEndianHost()
		{
			if (Endian::Native != Endian::Little)
			{
				throw NotSupportedException("PersistentStore R1 requires a little-endian host");
			}
		}

		static LoadedCandidate ReadCandidate(const FileBackend& backend,
		                                     const UInt64 slotOffset,
		                                     const UInt64 physicalFileBytes)
		{
			LoadedCandidate loaded;
			loaded.Candidate.SlotOffset = slotOffset;
			if (!IsRangeContained(0, physicalFileBytes, slotOffset, HeaderSlotBytes))
			{
				loaded.Candidate.RejectionReason = "header slot lies outside the physical file";
				return loaded;
			}

			try
			{
				std::array<Byte, HeaderSlotBytes> headerBytes{};
				backend.ReadExact(slotOffset, headerBytes.data(), headerBytes.size());
				loaded.Candidate.Header = ParseHeader(ConstByteView(headerBytes), physicalFileBytes);
				loaded.DirectoryBytes.resize(CheckedNarrow<std::size_t>(
				        loaded.Candidate.Header.DirectoryBytes, "directory read buffer size"));
				backend.ReadExact(loaded.Candidate.Header.DirectoryOffset,
				                  loaded.DirectoryBytes.data(), loaded.DirectoryBytes.size());
				loaded.Candidate.ParsedDirectory = ParseDirectory(
				        loaded.Candidate.Header, ConstByteView(loaded.DirectoryBytes), physicalFileBytes);
				loaded.Candidate.IsValid = true;
			}
			catch (const FormatException& exception)
			{
				loaded.Candidate.RejectionReason = exception.what();
			}
			return loaded;
		}

		static std::shared_ptr<StoreState> Load(FileBackend backend)
		{
			const UInt64 physicalFileBytes = backend.GetPhysicalFileBytes();
			LoadedCandidate slotA = ReadCandidate(backend, HeaderSlotAOffset, physicalFileBytes);
			LoadedCandidate slotB = ReadCandidate(backend, HeaderSlotBOffset, physicalFileBytes);

			if (!slotA.Candidate.IsValid && !slotB.Candidate.IsValid)
			{
				throw FormatException("both PersistentStore roots are invalid", slotA.Candidate.RejectionReason,
				                      slotB.Candidate.RejectionReason);
			}

			LoadedCandidate* selectedPtr = nullptr;
			if (!slotB.Candidate.IsValid
			    || (slotA.Candidate.IsValid
			        && slotA.Candidate.Header.Generation >= slotB.Candidate.Header.Generation))
			{
				selectedPtr = &slotA;
			}
			else
			{
				selectedPtr = &slotB;
			}

			if (slotA.Candidate.IsValid && slotB.Candidate.IsValid
			    && slotA.Candidate.Header.Generation == slotB.Candidate.Header.Generation
			    && (slotA.Candidate.Header.DirectoryOffset != slotB.Candidate.Header.DirectoryOffset
			        || slotA.Candidate.Header.DirectoryBytes != slotB.Candidate.Header.DirectoryBytes
			        || slotA.Candidate.Header.CommittedLogicalFileBytes
			                   != slotB.Candidate.Header.CommittedLogicalFileBytes
			        || slotA.Candidate.Header.DirectoryChecksum != slotB.Candidate.Header.DirectoryChecksum
			        || slotA.DirectoryBytes != slotB.DirectoryBytes))
			{
				throw FormatException("equal-generation PersistentStore roots contain different directories");
			}

			auto currentDirectoryPtr = std::make_shared<const Directory>(selectedPtr->Candidate.ParsedDirectory);
			return std::shared_ptr<StoreState>(new StoreState(
			        std::move(backend), std::move(slotA.Candidate), std::move(slotB.Candidate),
			        std::move(currentDirectoryPtr)));
		}

		static std::shared_ptr<StoreState> InitializeCreated(const std::filesystem::path& path,
		                                                         FileBackend backend)
		{
			try
			{
				Directory directory;
				directory.Generation = 1;
				const std::vector<Byte> directoryBytes = SerializeDirectory(directory);
				HeaderSlot header;
				header.Generation = 1;
				header.DirectoryOffset = SuperblockBytes;
				header.DirectoryBytes = directoryBytes.size();
				header.CommittedLogicalFileBytes = CheckedAdd(
				        SuperblockBytes, directoryBytes.size(), "initial PersistentStore length");
				header.DirectoryChecksum = FormatCodecDetail::ComputeCrc32C(ConstByteView(directoryBytes));
				const auto headerBytes = SerializeHeader(header);

				backend.SetPhysicalFileBytes(header.CommittedLogicalFileBytes);
				backend.WriteExact(header.DirectoryOffset, directoryBytes.data(), directoryBytes.size());
				backend.WriteExact(HeaderSlotAOffset, headerBytes.data(), headerBytes.size());

				RootCandidate slotA;
				slotA.IsValid = true;
				slotA.SlotOffset = HeaderSlotAOffset;
				slotA.Header = header;
				slotA.ParsedDirectory = directory;
				RootCandidate slotB;
				slotB.SlotOffset = HeaderSlotBOffset;
				slotB.RejectionReason = "unpublished header slot";
				auto currentDirectoryPtr = std::make_shared<const Directory>(directory);
				return std::shared_ptr<StoreState>(new StoreState(
				        std::move(backend), std::move(slotA), std::move(slotB),
				        std::move(currentDirectoryPtr)));
			}
			catch (...)
			{
				const bool isSameCreatedFile = backend.DoesPathIdentifySameFile(path);
				// Close first so Windows can remove the exclusively created file.
				{
					FileBackend closingBackend = std::move(backend);
				}
				if (isSameCreatedFile)
				{
					std::error_code ignoredError;
					std::filesystem::remove(path, ignoredError);
				}
				throw;
			}
		}

		FileBackend m_Backend;
		RootCandidate m_SlotA;
		RootCandidate m_SlotB;
		std::shared_ptr<const Directory> m_CurrentDirectoryPtr;
	};
}
