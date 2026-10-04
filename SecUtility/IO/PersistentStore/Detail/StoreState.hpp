// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Adapter.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FileBackend.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FormatCodec.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Mapping.hpp>
#include <SecUtility/Misc/Endian.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>


namespace SecUtility::IO::PersistentStoreDetail
{
	class StateMutex final
	{
	public:
		void lock() noexcept
		{
			while (m_Flag.test_and_set(std::memory_order_acquire))
			{
				std::this_thread::yield();
			}
		}

		void unlock() noexcept { m_Flag.clear(std::memory_order_release); }

	private:
		std::atomic_flag m_Flag = ATOMIC_FLAG_INIT;
	};

	class StateLockGuard final
	{
	public:
		explicit StateLockGuard(StateMutex& ref_mutex) noexcept : m_Mutex(ref_mutex) { m_Mutex.lock(); }
		~StateLockGuard() noexcept { m_Mutex.unlock(); }
		StateLockGuard(const StateLockGuard&) = delete;
		StateLockGuard& operator=(const StateLockGuard&) = delete;

	private:
		StateMutex& m_Mutex;
	};

	class StoreState final : public std::enable_shared_from_this<StoreState>
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
			StateLockGuard lock(m_StateMutex);
			return FindEntry(*m_CurrentDirectoryPtr, key) != nullptr;
		}

		std::size_t Size() const
		{
			StateLockGuard lock(m_StateMutex);
			return m_CurrentDirectoryPtr->Entries.size();
		}
		FileAccess GetAccess() const noexcept { return m_Backend.GetAccess(); }

		const RootCandidate& GetSlotA() const noexcept { return m_SlotA; }
		const RootCandidate& GetSlotB() const noexcept { return m_SlotB; }

		template <typename T, typename TValue>
		void Insert(const std::string_view key, TValue&& value)
		{
			Mutate<T>(key, std::forward<TValue>(value), MutationKind::Insert);
		}

		template <typename T, typename TValue>
		void InsertOrReassign(const std::string_view key, TValue&& value)
		{
			Mutate<T>(key, std::forward<TValue>(value), MutationKind::InsertOrReassign);
		}

		template <typename T, typename TValue>
		void Reassign(const std::string_view key, TValue&& value)
		{
			Mutate<T>(key, std::forward<TValue>(value), MutationKind::Reassign);
		}

		template <typename T>
		T Get(const std::string_view key) const
		{
			ValidateKey(key);
			DirectoryEntry entry;
			{
				StateLockGuard lock(m_StateMutex);
				const DirectoryEntry* const entryPtr = FindEntry(*m_CurrentDirectoryPtr, key);
				if (entryPtr == nullptr)
				{
					throw KeyNotFoundException("PersistentStore key was not found", key);
				}
				entry = *entryPtr;
			}
			const auto region = m_Backend.MapReadOnly(
			        entry.PayloadOffset, CheckedNarrow<std::size_t>(entry.PayloadBytes, "payload mapping size"), 1);
			return PersistentTraits<T>::Decode(ConstByteView(region.Data(), region.Size()));
		}

		template <typename T>
		auto GetLeased(const std::string_view key) const -> typename PersistentTraits<T>::LeasedType
		{
			ValidateKey(key);
			DirectoryEntry entry;
			{
				StateLockGuard lock(m_StateMutex);
				const DirectoryEntry* const entryPtr = FindEntry(*m_CurrentDirectoryPtr, key);
				if (entryPtr == nullptr)
				{
					throw KeyNotFoundException("PersistentStore key was not found", key);
				}
				entry = *entryPtr;
			}
			auto region = m_Backend.MapReadOnly(
			        entry.PayloadOffset, CheckedNarrow<std::size_t>(entry.PayloadBytes, "payload mapping size"), 1);
			auto leasePtr = std::make_shared<MappingLease>(shared_from_this(), std::move(region));
			return PersistentTraits<T>::DecodeLeased(std::move(leasePtr));
		}

	private:
		enum class MutationKind
		{
			Insert,
			InsertOrReassign,
			Reassign
		};

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

		static const DirectoryEntry* FindEntry(const Directory& directory, const std::string_view key)
		{
			const auto iterator = std::lower_bound(
			        directory.Entries.begin(), directory.Entries.end(), key,
			        [](const DirectoryEntry& entry, const std::string_view soughtKey)
			        { return FormatCodecDetail::IsUnsignedByteLess(entry.Key, soughtKey); });
			return iterator != directory.Entries.end() && iterator->Key.size() == key.size()
			               && std::equal(iterator->Key.begin(), iterator->Key.end(), key.begin())
			       ? &*iterator
			       : nullptr;
		}

		void RequireMutable() const
		{
			if (m_Backend.GetAccess() != FileAccess::ReadWrite)
			{
				throw InvalidOperationException("cannot mutate a read-only PersistentStore");
			}
			if (m_IsPoisoned)
			{
				throw InvalidOperationException("PersistentStore mutations are disabled after an indeterminate commit");
			}
		}

		static void ValidateMutationPrecondition(const Directory& directory,
		                                         const std::string_view key,
		                                         const MutationKind kind)
		{
			const bool hasKey = FindEntry(directory, key) != nullptr;
			if (kind == MutationKind::Insert && hasKey)
			{
				throw KeyAlreadyExistsException("PersistentStore key already exists", key);
			}
			if (kind == MutationKind::Reassign && !hasKey)
			{
				throw KeyNotFoundException("PersistentStore key was not found", key);
			}
		}

		template <typename T, typename TValue>
		void Mutate(const std::string_view key, TValue&& value, const MutationKind kind)
		{
			ValidateKey(key);
			{
				StateLockGuard stateLock(m_StateMutex);
				RequireMutable();
				ValidateMutationPrecondition(*m_CurrentDirectoryPtr, key, kind);
			}
			std::lock_guard<std::mutex> writerLock(m_WriterMutex);
			std::shared_ptr<const Directory> currentDirectoryPtr;
			{
				StateLockGuard stateLock(m_StateMutex);
				RequireMutable();
				currentDirectoryPtr = m_CurrentDirectoryPtr;
				ValidateMutationPrecondition(*currentDirectoryPtr, key, kind);
			}

			const PayloadLayout layout = PersistentTraits<T>::Measure(value);
			if (layout.Alignment == 0 || layout.Alignment > MaximumPayloadAlignment
			    || (layout.Alignment & (layout.Alignment - 1)) != 0 || layout.Bytes > MaximumFileBytes)
			{
				throw FormatException("adapter requested an invalid PersistentStore payload layout");
			}

			const UInt64 allocationOffset = m_Backend.GetPhysicalFileBytes();
			const UInt64 payloadOffset = CheckedAlignUp(allocationOffset, layout.Alignment, "payload alignment");
			const UInt64 storedBytes = std::max<UInt64>(layout.Bytes, 1);
			const UInt64 allocationCapacity = CheckedAdd(payloadOffset - allocationOffset, storedBytes,
			                                                   "payload allocation capacity");
			const UInt64 afterPayload = CheckedAdd(allocationOffset, allocationCapacity, "payload allocation end");
			if (afterPayload > MaximumFileBytes)
			{
				throw IOException("PersistentStore maximum file length exceeded");
			}
			m_Backend.SetPhysicalFileBytes(afterPayload);
			{
				auto region = m_Backend.MapReadWrite(
				        payloadOffset, CheckedNarrow<std::size_t>(layout.Bytes, "payload mapping size"), layout.Alignment);
				PersistentTraits<T>::Encode(std::forward<TValue>(value), MutableByteView(region.Data(), region.Size()));
				region.Complete();
			}

			Directory replacement = *currentDirectoryPtr;
			replacement.Generation = CheckedAdd(replacement.Generation, 1, "directory generation");
			DirectoryEntry replacementEntry{std::string(key), {allocationOffset, allocationCapacity}, payloadOffset,
			                                    layout.Bytes};
			auto iterator = std::lower_bound(
			        replacement.Entries.begin(), replacement.Entries.end(), key,
			        [](const DirectoryEntry& entry, const std::string_view soughtKey)
			        { return FormatCodecDetail::IsUnsignedByteLess(entry.Key, soughtKey); });
			if (iterator != replacement.Entries.end() && iterator->Key.size() == key.size()
			    && std::equal(iterator->Key.begin(), iterator->Key.end(), key.begin()))
			{
				*iterator = std::move(replacementEntry);
			}
			else
			{
				replacement.Entries.insert(iterator, std::move(replacementEntry));
			}

			const std::vector<Byte> directoryBytes = SerializeDirectory(replacement);
			const UInt64 directoryOffset = afterPayload;
			const UInt64 committedBytes = CheckedAdd(directoryOffset, directoryBytes.size(), "commit file length");
			if (committedBytes > MaximumFileBytes)
			{
				throw IOException("PersistentStore maximum file length exceeded");
			}
			m_Backend.SetPhysicalFileBytes(committedBytes);
			m_Backend.WriteExact(directoryOffset, directoryBytes.data(), directoryBytes.size());
			std::vector<Byte> checkedDirectoryBytes(directoryBytes.size());
			m_Backend.ReadExact(directoryOffset, checkedDirectoryBytes.data(), checkedDirectoryBytes.size());
			if (checkedDirectoryBytes != directoryBytes)
			{
				throw IOException("PersistentStore directory read-back mismatch before publication");
			}

			HeaderSlot header;
			header.Generation = replacement.Generation;
			header.DirectoryOffset = directoryOffset;
			header.DirectoryBytes = directoryBytes.size();
			header.CommittedLogicalFileBytes = committedBytes;
			header.DirectoryChecksum = FormatCodecDetail::ComputeCrc32C(ConstByteView(directoryBytes));
			const auto headerBytes = SerializeHeader(header);
			(void)ParseDirectory(header, ConstByteView(checkedDirectoryBytes), committedBytes);
			auto replacementDirectoryPtr = std::make_shared<const Directory>(replacement);
			RootCandidate replacementRoot;
			replacementRoot.IsValid = true;
			replacementRoot.Header = header;
			replacementRoot.ParsedDirectory = replacement;

			RootCandidate* targetPtr = !m_SlotA.IsValid ? &m_SlotA
			                              : !m_SlotB.IsValid ? &m_SlotB
			                              : m_SlotA.Header.Generation <= m_SlotB.Header.Generation ? &m_SlotA : &m_SlotB;
			const UInt64 slotOffset = targetPtr == &m_SlotA ? HeaderSlotAOffset : HeaderSlotBOffset;
			replacementRoot.SlotOffset = slotOffset;
			try
			{
				m_Backend.WriteExact(slotOffset, headerBytes.data(), headerBytes.size());
				std::array<Byte, HeaderSlotBytes> checkedHeaderBytes{};
				m_Backend.ReadExact(slotOffset, checkedHeaderBytes.data(), checkedHeaderBytes.size());
				const HeaderSlot checkedHeader = ParseHeader(ConstByteView(checkedHeaderBytes), committedBytes);
				(void)ParseDirectory(checkedHeader, ConstByteView(checkedDirectoryBytes), committedBytes);
			}
			catch (...)
			{
				StateLockGuard stateLock(m_StateMutex);
				m_IsPoisoned = true;
				throw IOException("PersistentStore commit result is uncertain; release all leases, reopen the store, and reread the affected key");
			}

			{
				static_assert(std::is_nothrow_swappable_v<RootCandidate>);
				StateLockGuard stateLock(m_StateMutex);
				m_CurrentDirectoryPtr.swap(replacementDirectoryPtr);
				using std::swap;
				swap(*targetPtr, replacementRoot);
			}
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
		mutable std::mutex m_WriterMutex;
		mutable StateMutex m_StateMutex;
		bool m_IsPoisoned = false;
		RootCandidate m_SlotA;
		RootCandidate m_SlotB;
		std::shared_ptr<const Directory> m_CurrentDirectoryPtr;
	};
}
