// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

// ReSharper disable CppUseDesignatedInitializers
#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Adapter.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FileBackend.hpp>
#include <SecUtility/IO/PersistentStore/Detail/ExtentAllocator.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FormatCodec.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Mapping.hpp>
#include <SecUtility/Misc/Endian.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <map>
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
		class AdapterCallbackGuard final
		{
		public:
			explicit AdapterCallbackGuard(const StoreState* const storePtr)
			    : m_StorePtr(storePtr), m_PreviousPtr(ActivePtr)
			{
				RejectReentry(storePtr);
				ActivePtr = this;
			}
			~AdapterCallbackGuard() noexcept { ActivePtr = m_PreviousPtr; }
			AdapterCallbackGuard(const AdapterCallbackGuard&) = delete;
			AdapterCallbackGuard& operator=(const AdapterCallbackGuard&) = delete;

			static void RejectReentry(const StoreState* const storePtr)
			{
				for (const AdapterCallbackGuard* framePtr = ActivePtr; framePtr != nullptr;
				     framePtr = framePtr->m_PreviousPtr)
				{
					if (framePtr->m_StorePtr == storePtr)
						throw InvalidOperationException(
						        "adapter callback reentry into the same PersistentStore is prohibited");
				}
			}

		private:
			inline static thread_local AdapterCallbackGuard* ActivePtr = nullptr;
			const StoreState* m_StorePtr;
			AdapterCallbackGuard* m_PreviousPtr;
		};

		struct PinnedPayload
		{
			Extent AllocatedExtent;
			UInt64 PayloadOffset = 0;
			UInt64 PayloadBytes = 0;
		};

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
			AdapterCallbackGuard::RejectReentry(this);
			ValidateKey(key);
			StateLockGuard lock(m_StateMutex);
			return FindEntry(*m_CurrentDirectoryPtr, key) != nullptr;
		}

		std::size_t Size() const
		{
			AdapterCallbackGuard::RejectReentry(this);
			StateLockGuard lock(m_StateMutex);
			return m_CurrentDirectoryPtr->Entries.size();
		}
		FileAccess GetAccess() const noexcept { return m_Backend.GetAccess(); }

		// Direct root references are intended only for single-threaded diagnostics/tests. Concurrent
		// mutation can replace either object; these accessors may be removed in a future revision.
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
			AdapterCallbackGuard::RejectReentry(this);
			ValidateKey(key);
			const PinnedPayload entry = FindAndPinPayload(key);
			const auto leasePtr = MapPinned(entry);
			AdapterCallbackGuard callbackGuard(this);
			return PersistentTraits<T>::Decode(ConstByteView(leasePtr->Data(), leasePtr->Size()));
		}

		template <typename T>
		auto GetLeased(const std::string_view key) const -> typename PersistentTraits<T>::LeasedType
		{
			AdapterCallbackGuard::RejectReentry(this);
			ValidateKey(key);
			const PinnedPayload entry = FindAndPinPayload(key);
			auto leasePtr = MapPinned(entry);
			AdapterCallbackGuard callbackGuard(this);
			return PersistentTraits<T>::DecodeLeased(std::move(leasePtr));
		}

		bool Erase(const std::string_view key)
		{
			AdapterCallbackGuard::RejectReentry(this);
			ValidateKey(key);
			std::lock_guard<std::mutex> writerLock(m_WriterMutex);
			std::shared_ptr<const Directory> currentDirectoryPtr;
			{
				StateLockGuard stateLock(m_StateMutex);
				RequireMutable();
				currentDirectoryPtr = m_CurrentDirectoryPtr;
				if (FindEntry(*currentDirectoryPtr, key) == nullptr) return false;
			}
			RebuildAllocatorForWriter();

			Directory replacement = *currentDirectoryPtr;
			const auto iterator = std::lower_bound(
			        replacement.Entries.begin(), replacement.Entries.end(), key,
			        [](const DirectoryEntry& entry, const std::string_view soughtKey)
			        { return FormatCodecDetail::IsUnsignedByteLess(entry.Key, soughtKey); });
			replacement.Entries.erase(iterator);
			replacement.Generation = CheckedAdd(replacement.Generation, 1, "directory generation");
			PublishDirectory(std::move(replacement));
			return true;
		}

		void ReleasePin(const ExtentIdentity identity) const noexcept
		{
			std::map<ExtentIdentity, std::size_t>::node_type releasedNode;
			{
				StateLockGuard lock(m_StateMutex);
				const auto iterator = m_PinCounts.find(identity);
				if (iterator == m_PinCounts.end()) return;
				if (--iterator->second == 0) releasedNode = m_PinCounts.extract(iterator);
			}
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
			RebuildAllocatorForWriter();
		}

		PinnedPayload FindAndPinPayload(const std::string_view key) const
		{
			std::map<ExtentIdentity, std::size_t> preparedNodeOwner;
			preparedNodeOwner.emplace(ExtentIdentity{}, 1);
			auto preparedNode = preparedNodeOwner.extract(preparedNodeOwner.begin());
			StateLockGuard lock(m_StateMutex);
			const DirectoryEntry* const entryPtr = FindEntry(*m_CurrentDirectoryPtr, key);
			if (entryPtr == nullptr)
				throw KeyNotFoundException("PersistentStore key was not found", key);
			const ExtentIdentity identity{entryPtr->AllocatedExtent.Offset, entryPtr->AllocatedExtent.Capacity};
			const auto iterator = m_PinCounts.find(identity);
			if (iterator == m_PinCounts.end())
			{
				preparedNode.key() = identity;
				m_PinCounts.insert(std::move(preparedNode));
			}
			else
			{
				++iterator->second;
			}
			return {entryPtr->AllocatedExtent, entryPtr->PayloadOffset, entryPtr->PayloadBytes};
		}

		std::shared_ptr<MappingLease> MapPinned(const PinnedPayload& entry) const
		{
			const ExtentIdentity identity{entry.AllocatedExtent.Offset, entry.AllocatedExtent.Capacity};
			try
			{
				auto region = m_Backend.MapReadOnly(
				        entry.PayloadOffset, CheckedNarrow<std::size_t>(entry.PayloadBytes, "payload mapping size"), 1);
				return std::make_shared<MappingLease>(shared_from_this(), std::move(region), identity);
			}
			catch (...)
			{
				ReleasePin(identity);
				throw;
			}
		}

		static void AppendRootExtents(const RootCandidate& root, std::vector<Extent>& ref_extents)
		{
			if (!root.IsValid) return;
			ref_extents.push_back({root.Header.DirectoryOffset, root.Header.DirectoryBytes});
			for (const DirectoryEntry& entry : root.ParsedDirectory.Entries)
				ref_extents.push_back(entry.AllocatedExtent);
		}

		std::vector<Extent> CollectProtectedExtentsForWriter() const
		{
			std::vector<Extent> extents;
			extents.reserve(m_SlotA.ParsedDirectory.Entries.size() + m_SlotB.ParsedDirectory.Entries.size() + 2);
			AppendRootExtents(m_SlotA, extents);
			AppendRootExtents(m_SlotB, extents);
			const std::size_t rootExtentCount = extents.size();
			while (true)
			{
				std::size_t pinExtentCount = 0;
				{
					StateLockGuard stateLock(m_StateMutex);
					pinExtentCount = m_PinCounts.size();
				}
				if (extents.capacity() < rootExtentCount + pinExtentCount)
					extents.reserve(rootExtentCount + pinExtentCount);
				extents.resize(rootExtentCount);
				StateLockGuard stateLock(m_StateMutex);
				if (m_PinCounts.size() > extents.capacity() - rootExtentCount) continue;
				for (const auto& [identity, count] : m_PinCounts)
				{
					(void)count;
					extents.push_back({identity.Offset, identity.Capacity});
				}
				return extents;
			}
		}

		void RebuildAllocatorForWriter()
		{
			ScopedFailureInjection::Observe(FailurePoint::RebuildAllocator);
			std::vector<Extent> protectedExtents = CollectProtectedExtentsForWriter();
			const UInt64 physicalBytes = m_Backend.GetPhysicalFileBytes();
			ExtentAllocator replacement;
			replacement.Rebuild(physicalBytes, std::move(protectedExtents));
			using std::swap;
			swap(m_Allocator, replacement);
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

		struct ClassifiedExtent
		{
			Extent Range;
			const DirectoryEntry* EntryPtr = nullptr;
			unsigned int RootIndex = 0;
		};

		static bool AreSharedEntriesConsistent(const DirectoryEntry& left, const DirectoryEntry& right)
		{
			return left.Key == right.Key && left.AllocatedExtent.Offset == right.AllocatedExtent.Offset
			       && left.AllocatedExtent.Capacity == right.AllocatedExtent.Capacity
			       && left.PayloadOffset == right.PayloadOffset && left.PayloadBytes == right.PayloadBytes;
		}

		static void ValidateCrossRootExtents(const RootCandidate& slotA, const RootCandidate& slotB)
		{
			if (!slotA.IsValid || !slotB.IsValid) return;
			std::vector<ClassifiedExtent> extents;
			extents.reserve(slotA.ParsedDirectory.Entries.size() + slotB.ParsedDirectory.Entries.size() + 2);
			extents.push_back({{slotA.Header.DirectoryOffset, slotA.Header.DirectoryBytes}, nullptr, 0});
			extents.push_back({{slotB.Header.DirectoryOffset, slotB.Header.DirectoryBytes}, nullptr, 1});
			for (const DirectoryEntry& entry : slotA.ParsedDirectory.Entries)
				extents.push_back({entry.AllocatedExtent, &entry, 0});
			for (const DirectoryEntry& entry : slotB.ParsedDirectory.Entries)
				extents.push_back({entry.AllocatedExtent, &entry, 1});
			std::sort(extents.begin(), extents.end(), [](const ClassifiedExtent& left, const ClassifiedExtent& right)
			{
				return left.Range.Offset < right.Range.Offset
				       || (left.Range.Offset == right.Range.Offset && left.Range.Capacity < right.Range.Capacity);
			});
			for (std::size_t index = 1; index < extents.size(); ++index)
			{
				const ClassifiedExtent& previous = extents[index - 1];
				const ClassifiedExtent& current = extents[index];
				if (previous.RootIndex == current.RootIndex
				    || !DoRangesOverlap(previous.Range.Offset, previous.Range.Capacity,
				                        current.Range.Offset, current.Range.Capacity))
					continue;
				const bool isConsistentSharedPayload = previous.EntryPtr != nullptr && current.EntryPtr != nullptr
				        && AreSharedEntriesConsistent(*previous.EntryPtr, *current.EntryPtr);
				const bool isConsistentSharedDirectory = previous.EntryPtr == nullptr && current.EntryPtr == nullptr
				        && previous.Range.Offset == current.Range.Offset
				        && previous.Range.Capacity == current.Range.Capacity
				        && slotA.Header.Generation == slotB.Header.Generation
				        && slotA.Header.DirectoryChecksum == slotB.Header.DirectoryChecksum;
				if (!isConsistentSharedPayload && !isConsistentSharedDirectory)
					throw FormatException("PersistentStore roots contain inconsistent overlapping extents");
			}
		}

		static std::shared_ptr<StoreState> Load(FileBackend backend)
		{
			const UInt64 physicalFileBytes = backend.GetPhysicalFileBytes();
			LoadedCandidate slotA = ReadCandidate(backend, HeaderSlotAOffset, physicalFileBytes);
			LoadedCandidate slotB = ReadCandidate(backend, HeaderSlotBOffset, physicalFileBytes);
			ValidateCrossRootExtents(slotA.Candidate, slotB.Candidate);

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
			AdapterCallbackGuard::RejectReentry(this);
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
			RebuildAllocatorForWriter();

			PayloadLayout layout;
			{
				AdapterCallbackGuard callbackGuard(this);
				layout = PersistentTraits<T>::Measure(value);
			}
			if (layout.Alignment == 0 || layout.Alignment > MaximumPayloadAlignment
			    || (layout.Alignment & (layout.Alignment - 1)) != 0 || layout.Bytes > MaximumFileBytes)
			{
				throw FormatException("adapter requested an invalid PersistentStore payload layout");
			}

			ScopedFailureInjection::Observe(FailurePoint::ReservePayload);
			const Allocation payloadAllocation = m_Allocator.Reserve(layout.Bytes, layout.Alignment);
			if (m_Allocator.GetPhysicalBytes() > MaximumFileBytes)
			{
				throw IOException("PersistentStore maximum file length exceeded");
			}
			ScopedFailureInjection::Observe(FailurePoint::GrowPayload);
			m_Backend.SetPhysicalFileBytes(m_Allocator.GetPhysicalBytes());
			{
				ScopedFailureInjection::Observe(FailurePoint::MapPayload);
				auto region = m_Backend.MapReadWrite(
				        payloadAllocation.PayloadOffset,
				        CheckedNarrow<std::size_t>(layout.Bytes, "payload mapping size"), layout.Alignment);
				{
					AdapterCallbackGuard callbackGuard(this);
					ScopedFailureInjection::Observe(FailurePoint::EncodePayload);
					PersistentTraits<T>::Encode(
					        std::forward<TValue>(value), MutableByteView(region.Data(), region.Size()));
				}
				ScopedFailureInjection::Observe(FailurePoint::CompletePayload);
				region.Complete();
			}

			Directory replacement = *currentDirectoryPtr;
			replacement.Generation = CheckedAdd(replacement.Generation, 1, "directory generation");
			DirectoryEntry replacementEntry{std::string(key), payloadAllocation.AllocatedExtent,
			                                    payloadAllocation.PayloadOffset,
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

			PublishDirectory(std::move(replacement));
		}

		void PublishDirectory(Directory replacement)
		{
			ScopedFailureInjection::Observe(FailurePoint::SerializeDirectory);
			const std::vector<Byte> directoryBytes = SerializeDirectory(replacement);
			ScopedFailureInjection::Observe(FailurePoint::ReserveDirectory);
			const Allocation directoryAllocation = m_Allocator.Reserve(directoryBytes.size(), 1, false);
			const UInt64 directoryOffset = directoryAllocation.PayloadOffset;
			const UInt64 committedBytes = m_Allocator.GetPhysicalBytes();
			if (committedBytes > MaximumFileBytes)
			{
				throw IOException("PersistentStore maximum file length exceeded");
			}
			ScopedFailureInjection::Observe(FailurePoint::GrowDirectory);
			m_Backend.SetPhysicalFileBytes(committedBytes);
			ScopedFailureInjection::Observe(FailurePoint::WriteDirectory);
			m_Backend.WriteExact(directoryOffset, directoryBytes.data(), directoryBytes.size());
			std::vector<Byte> checkedDirectoryBytes(directoryBytes.size());
			ScopedFailureInjection::Observe(FailurePoint::ReadBackDirectory);
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
			ScopedFailureInjection::Observe(FailurePoint::PrepareRuntimeState);
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
				ScopedFailureInjection::Observe(FailurePoint::BeginPublication);
				if (ScopedFailureInjection::IsEnabled())
				{
					for (std::size_t index = 0; index < headerBytes.size(); ++index)
					{
						m_Backend.WriteExact(slotOffset + index, headerBytes.data() + index, 1);
						ScopedFailureInjection::Observe(FailurePoint::PublicationWriteProgress, index + 1);
					}
				}
				else
				{
					m_Backend.WriteExact(slotOffset, headerBytes.data(), headerBytes.size());
				}
				std::array<Byte, HeaderSlotBytes> checkedHeaderBytes{};
				ScopedFailureInjection::Observe(FailurePoint::ReadBackPublication);
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
		ExtentAllocator m_Allocator;
		mutable std::map<ExtentIdentity, std::size_t> m_PinCounts;
	};

	inline MappingLease::~MappingLease() noexcept
	{
		m_StatePtr->ReleasePin(m_Identity);
	}
}
