// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#if defined(_WIN32)
#error PosixFileBackend.hpp cannot be used on Windows
#endif

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/IO/PersistentStore/Detail/CheckedArithmetic.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Format.hpp>
#include <SecUtility/Raw/Int.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


namespace SecUtility::IO::PersistentStoreDetail
{
	template <bool IsWritable>
	class PosixMappedRegion final
	{
	public:
		PosixMappedRegion() noexcept = default;

		PosixMappedRegion(void* const basePtr,
		                  const std::size_t mappedBytes,
		                  Byte* const exposedPtr,
		                  const std::size_t exposedBytes) noexcept
		    : m_BasePtr(basePtr), m_MappedBytes(mappedBytes), m_ExposedPtr(exposedPtr), m_ExposedBytes(exposedBytes)
		{
			/* NO CODE */
		}

		PosixMappedRegion(PosixMappedRegion&& other) noexcept
		    : m_BasePtr(std::exchange(other.m_BasePtr, nullptr)), m_MappedBytes(std::exchange(other.m_MappedBytes, 0)),
		      m_ExposedPtr(std::exchange(other.m_ExposedPtr, nullptr)),
		      m_ExposedBytes(std::exchange(other.m_ExposedBytes, 0))
		{
			/* NO CODE */
		}

		PosixMappedRegion& operator=(PosixMappedRegion&& other) noexcept
		{
			if (this != &other)
			{
				Reset();
				m_BasePtr = std::exchange(other.m_BasePtr, nullptr);
				m_MappedBytes = std::exchange(other.m_MappedBytes, 0);
				m_ExposedPtr = std::exchange(other.m_ExposedPtr, nullptr);
				m_ExposedBytes = std::exchange(other.m_ExposedBytes, 0);
			}
			return *this;
		}

		PosixMappedRegion(const PosixMappedRegion&) = delete;
		PosixMappedRegion& operator=(const PosixMappedRegion&) = delete;
		~PosixMappedRegion() noexcept
		{
			Reset();
		}

		auto Data() const noexcept
		{
			if constexpr (IsWritable)
			{
				return m_ExposedPtr;
			}
			else
			{
				return static_cast<const Byte*>(m_ExposedPtr);
			}
		}

		std::size_t Size() const noexcept
		{
			return m_ExposedBytes;
		}

		template <bool IsEnabled = IsWritable, std::enable_if_t<IsEnabled, int> = 0>
		void Complete()
		{
			if (m_BasePtr == nullptr)
			{
				return;
			}
			if (::msync(m_BasePtr, m_MappedBytes, MS_SYNC) != 0)
			{
				throw IOException(std::string("msync: ") + std::generic_category().message(errno));
			}
			if (::munmap(m_BasePtr, m_MappedBytes) != 0)
			{
				throw IOException(std::string("munmap: ") + std::generic_category().message(errno));
			}
			m_BasePtr = nullptr;
			m_MappedBytes = 0;
			m_ExposedPtr = nullptr;
			m_ExposedBytes = 0;
		}

	private:
		void Reset() noexcept
		{
			if (m_BasePtr != nullptr)
			{
				::munmap(m_BasePtr, m_MappedBytes);
			}
			m_BasePtr = nullptr;
			m_MappedBytes = 0;
			m_ExposedPtr = nullptr;
			m_ExposedBytes = 0;
		}

		void* m_BasePtr = nullptr;
		std::size_t m_MappedBytes = 0;
		Byte* m_ExposedPtr = nullptr;
		std::size_t m_ExposedBytes = 0;
	};

	using PosixReadOnlyMappedRegion = PosixMappedRegion<false>;
	using PosixReadWriteMappedRegion = PosixMappedRegion<true>;

	struct PosixFileIdentity
	{
		::dev_t Device = 0;
		::ino_t Inode = 0;

		friend bool operator==(const PosixFileIdentity& left, const PosixFileIdentity& right) noexcept
		{
			return left.Device == right.Device && left.Inode == right.Inode;
		}
	};

	struct PosixFileIdentityHash
	{
		std::size_t operator()(const PosixFileIdentity& identity) const noexcept
		{
			const std::size_t first = std::hash<dev_t>{}(identity.Device);
			const std::size_t second = std::hash<ino_t>{}(identity.Inode);
			return first ^ (second + 0x9E3779B9U + (first << 6U) + (first >> 2U));
		}
	};

	namespace PosixFileBackendDetail
	{
		struct RegistryEntry
		{
			std::size_t ReaderCount = 0;
			bool HasWriter = false;
		};

		inline std::mutex RegistryMutex;
		inline std::unordered_map<PosixFileIdentity, RegistryEntry, PosixFileIdentityHash> Registry;

		inline std::string ErrorMessage(const char* operation, const int error)
		{
			return std::string(operation) + ": " + std::generic_category().message(error);
		}

		inline void AcquireRegistry(const PosixFileIdentity& identity, const FileAccess access)
		{
			std::lock_guard<std::mutex> lock(RegistryMutex);
			auto iterator = Registry.find(identity);
			if (iterator == Registry.end())
			{
				iterator = Registry.emplace(identity, RegistryEntry{}).first;
			}

			auto& [readerCount, hasWriter] = iterator->second;
			if (const bool hasConflict = access == FileAccess::ReadOnly ? hasWriter : hasWriter || readerCount != 0;
			    hasConflict)
			{
				if (readerCount == 0 && !hasWriter)
				{
					Registry.erase(iterator);
				}
				throw IOException("PersistentStore lock unavailable in this process");
			}

			if (access == FileAccess::ReadOnly)
			{
				++readerCount;
			}
			else
			{
				hasWriter = true;
			}
		}

		inline void ReleaseRegistry(const PosixFileIdentity& identity, const FileAccess access) noexcept
		{
			std::lock_guard<std::mutex> lock(RegistryMutex);
			const auto iterator = Registry.find(identity);
			if (iterator == Registry.end())
			{
				return;
			}

			auto& [readerCount, hasWriter] = iterator->second;
			if (access == FileAccess::ReadOnly)
			{
				if (readerCount != 0)
				{
					--readerCount;
				}
			}
			else
			{
				hasWriter = false;
			}
			if (readerCount == 0 && !hasWriter)
			{
				Registry.erase(iterator);
			}
		}
	}

	class PosixFileBackend final
	{
	public:
		static PosixFileBackend Open(const std::filesystem::path& path, const FileAccess access)
		{
			const int flags = access == FileAccess::ReadOnly ? O_RDONLY : O_RDWR;
			const int descriptor = ::open(path.c_str(), flags | O_CLOEXEC);
			if (descriptor == -1)
			{
				throw IOException("PersistentStore open failed",
				                  path.string(),
				                  PosixFileBackendDetail::ErrorMessage("open", errno));
			}
			return AdoptAndLock(descriptor, access, path);
		}

		static std::optional<PosixFileBackend> TryCreateExclusive(const std::filesystem::path& path)
		{
			const int descriptor = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
			if (descriptor == -1)
			{
				if (errno == EEXIST)
				{
					return std::nullopt;
				}
				throw IOException("PersistentStore exclusive create failed",
				                  path.string(),
				                  PosixFileBackendDetail::ErrorMessage("open", errno));
			}
			return AdoptAndLock(descriptor, FileAccess::ReadWrite, path);
		}

		PosixFileBackend(PosixFileBackend&& other) noexcept
		    : m_Descriptor(std::exchange(other.m_Descriptor, -1)), m_Identity(other.m_Identity),
		      m_Access(other.m_Access), m_HasRegistryEntry(std::exchange(other.m_HasRegistryEntry, false))
		{
			/* NO CODE */
		}

		PosixFileBackend& operator=(PosixFileBackend&& other) noexcept
		{
			if (this != &other)
			{
				Close();
				m_Descriptor = std::exchange(other.m_Descriptor, -1);
				m_Identity = other.m_Identity;
				m_Access = other.m_Access;
				m_HasRegistryEntry = std::exchange(other.m_HasRegistryEntry, false);
			}
			return *this;
		}

		PosixFileBackend(const PosixFileBackend&) = delete;
		PosixFileBackend& operator=(const PosixFileBackend&) = delete;
		~PosixFileBackend() noexcept
		{
			Close();
		}

		UInt64 GetPhysicalFileBytes() const
		{
			struct stat status{};
			if (::fstat(m_Descriptor, &status) != 0)
			{
				throw IOException(PosixFileBackendDetail::ErrorMessage("fstat", errno));
			}
			if (status.st_size < 0)
			{
				throw IOException("PersistentStore file has a negative length");
			}
			return static_cast<UInt64>(status.st_size);
		}

		// ReSharper disable once CppMemberFunctionMayBeConst
		void SetPhysicalFileBytes(const UInt64 byteCount)
		{
			RequireWritable("grow file");
			if (byteCount > static_cast<UInt64>(std::numeric_limits<off_t>::max()))
			{
				throw IOException("PersistentStore file length exceeds the POSIX offset range", byteCount);
			}
			if (::ftruncate(m_Descriptor, static_cast<off_t>(byteCount)) != 0)
			{
				throw IOException(PosixFileBackendDetail::ErrorMessage("ftruncate", errno));
			}
		}

		void ReadExact(const UInt64 offset, Byte* out_bytesPtr, const std::size_t byteCount) const
		{
			ValidateBufferAndRange(offset, out_bytesPtr, byteCount, "read");
			CompleteExactTransfer(
			        offset,
			        out_bytesPtr,
			        byteCount,
			        true,
			        [this](const UInt64 currentOffset, Byte* currentBytesPtr, const std::size_t remainingBytes)
			        {
				        const std::size_t chunk = std::min<std::size_t>(
				                remainingBytes, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
				        const ssize_t result =
				                ::pread(m_Descriptor, currentBytesPtr, chunk, static_cast<off_t>(currentOffset));
				        if (result > 0)
				        {
					        return ExactTransferResult{static_cast<std::size_t>(result), false};
				        }
				        if (result == 0)
				        {
					        return ExactTransferResult{};
				        }
				        if (errno == EINTR)
				        {
					        return ExactTransferResult{0, true};
				        }
				        throw IOException(PosixFileBackendDetail::ErrorMessage("pread", errno));
			        });
		}

		// ReSharper disable once CppMemberFunctionMayBeConst
		void WriteExact(const UInt64 offset, const Byte* bytesPtr, const std::size_t byteCount)
		{
			RequireWritable("write file");
			ValidateBufferAndRange(offset, bytesPtr, byteCount, "write");
			CompleteExactTransfer(
			        offset,
			        bytesPtr,
			        byteCount,
			        false,
			        [this](const UInt64 currentOffset, const Byte* currentBytesPtr, const std::size_t remainingBytes)
			        {
				        const std::size_t chunk = std::min<std::size_t>(
				                remainingBytes, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
				        const ssize_t result =
				                ::pwrite(m_Descriptor, currentBytesPtr, chunk, static_cast<off_t>(currentOffset));
				        if (result > 0)
				        {
					        return ExactTransferResult{static_cast<std::size_t>(result), false};
				        }
				        if (result == 0)
				        {
					        return ExactTransferResult{};
				        }
				        if (errno == EINTR)
				        {
					        return ExactTransferResult{0, true};
				        }
				        throw IOException(PosixFileBackendDetail::ErrorMessage("pwrite", errno));
			        });
		}

		FileAccess GetAccess() const noexcept
		{
			return m_Access;
		}

		bool DoesPathIdentifySameFile(const std::filesystem::path& path) const noexcept
		{
			struct stat status{};
			return ::stat(path.c_str(), &status) == 0 && PosixFileIdentity{status.st_dev, status.st_ino} == m_Identity;
		}

		static UInt64 GetMappingGranularity()
		{
			const long pageBytes = ::sysconf(_SC_PAGE_SIZE);
			if (pageBytes <= 0)
			{
				throw IOException("PersistentStore could not query the POSIX page size");
			}
			return static_cast<UInt64>(pageBytes);
		}

		PosixReadOnlyMappedRegion MapReadOnly(const UInt64 offset,
		                                      const std::size_t byteCount,
		                                      const UInt64 alignment) const
		{
			return Map<false>(offset, byteCount, alignment);
		}

		PosixReadWriteMappedRegion MapReadWrite(const UInt64 offset,
		                                        const std::size_t byteCount,
		                                        const UInt64 alignment)
		{
			RequireWritable("map writable file region");
			return Map<true>(offset, byteCount, alignment);
		}

	private:
		template <bool IsWritable>
		PosixMappedRegion<IsWritable> Map(const UInt64 offset,
		                                  const std::size_t byteCount,
		                                  const UInt64 alignment) const
		{
			ValidateMappingRequest(offset, byteCount, alignment);
			if (byteCount == 0)
			{
				return {};
			}

			const UInt64 granularity = GetMappingGranularity();
			const UInt64 mappedOffset = offset - offset % granularity;
			const UInt64 delta = offset - mappedOffset;
			const UInt64 mappedBytes64 = CheckedAdd(delta, byteCount, "PersistentStore mapped byte count");
			const std::size_t mappedBytes =
			        CheckedNarrow<std::size_t>(mappedBytes64, "PersistentStore native mapped byte count");
			const int protection = IsWritable ? PROT_READ | PROT_WRITE : PROT_READ;
			void* const basePtr = ::mmap(
			        nullptr, mappedBytes, protection, MAP_SHARED, m_Descriptor, static_cast<off_t>(mappedOffset));
			if (basePtr == MAP_FAILED)
			{
				throw IOException(PosixFileBackendDetail::ErrorMessage("mmap", errno));
			}
			Byte* const exposedPtr = static_cast<Byte*>(basePtr) + static_cast<std::size_t>(delta);
			if (reinterpret_cast<std::uintptr_t>(exposedPtr) % alignment != 0)
			{
				::munmap(basePtr, mappedBytes);
				throw IOException("PersistentStore mapped address does not satisfy the requested alignment");
			}
			return {basePtr, mappedBytes, exposedPtr, byteCount};
		}

		void ValidateMappingRequest(const UInt64 offset, const std::size_t byteCount, const UInt64 alignment) const
		{
			if (alignment == 0 || alignment > MaximumPayloadAlignment || (alignment & (alignment - 1)) != 0)
			{
				throw IOException("PersistentStore mapping alignment must be a power of two from 1 through 4096");
			}
			if (offset % alignment != 0)
			{
				throw IOException("PersistentStore mapping offset does not satisfy the requested alignment");
			}
			if (offset > static_cast<UInt64>(std::numeric_limits<off_t>::max())
			    || byteCount > static_cast<UInt64>(std::numeric_limits<off_t>::max()) - offset)
			{
				throw IOException("PersistentStore mapping exceeds the POSIX offset range");
			}
			const UInt64 physicalFileBytes = GetPhysicalFileBytes();
			if (offset > physicalFileBytes || byteCount > physicalFileBytes - offset)
			{
				throw IOException("PersistentStore mapping extends past the physical file length");
			}
		}

		PosixFileBackend(const int descriptor, const PosixFileIdentity identity, const FileAccess access) noexcept
		    : m_Descriptor(descriptor), m_Identity(identity), m_Access(access), m_HasRegistryEntry(true)
		{
			/* NO CODE */
		}

		static PosixFileBackend AdoptAndLock(const int descriptor,
		                                     const FileAccess access,
		                                     const std::filesystem::path& path)
		{
			struct stat status{};
			if (::fstat(descriptor, &status) != 0)
			{
				const int error = errno;
				::close(descriptor);
				throw IOException("PersistentStore fstat failed",
				                  path.string(),
				                  PosixFileBackendDetail::ErrorMessage("fstat", error));
			}
			if (!S_ISREG(status.st_mode))
			{
				::close(descriptor);
				throw IOException("PersistentStore path is not a regular file", path.string());
			}

			const PosixFileIdentity identity{status.st_dev, status.st_ino};
			try
			{
				PosixFileBackendDetail::AcquireRegistry(identity, access);
			}
			catch (...)
			{
				::close(descriptor);
				throw;
			}

			if (!TryAcquireNativeLock(descriptor, access))
			{
				const int error = errno;
				PosixFileBackendDetail::ReleaseRegistry(identity, access);
				::close(descriptor);
				throw IOException("PersistentStore lock unavailable",
				                  path.string(),
				                  PosixFileBackendDetail::ErrorMessage("lock", error));
			}
			return PosixFileBackend(descriptor, identity, access);
		}

		static bool TryAcquireNativeLock(const int descriptor, const FileAccess access) noexcept
		{
#if defined(F_OFD_SETLK) && defined(__linux__)
			struct flock lock{};
			lock.l_type = access == FileAccess::ReadOnly ? F_RDLCK : F_WRLCK;
			lock.l_whence = SEEK_SET;
			lock.l_start = 0;
			lock.l_len = 0;
			return ::fcntl(descriptor, F_OFD_SETLK, &lock) == 0;
#else
			const int operation = (access == FileAccess::ReadOnly ? LOCK_SH : LOCK_EX) | LOCK_NB;
			return ::flock(descriptor, operation) == 0;
#endif
		}

		template <typename TByte>
		static void ValidateBufferAndRange(const UInt64 offset,
		                                   TByte* bytesPtr,
		                                   const std::size_t byteCount,
		                                   const char* operation)
		{
			if (bytesPtr == nullptr && byteCount != 0)
			{
				throw IOException("PersistentStore exact I/O received a null buffer", operation);
			}
			constexpr auto maximumOffset = static_cast<UInt64>(std::numeric_limits<off_t>::max());
			if (offset > maximumOffset || byteCount > maximumOffset - offset)
			{
				throw IOException("PersistentStore exact I/O exceeds the POSIX offset range", operation);
			}
		}

		void RequireWritable(const char* operation) const
		{
			if (m_Access != FileAccess::ReadWrite)
			{
				throw InvalidOperationException("cannot perform a write operation through a read-only PersistentStore",
				                                operation);
			}
		}

		void Close() noexcept
		{
			if (m_Descriptor != -1)
			{
				::close(m_Descriptor);
				m_Descriptor = -1;
			}
			if (m_HasRegistryEntry)
			{
				PosixFileBackendDetail::ReleaseRegistry(m_Identity, m_Access);
				m_HasRegistryEntry = false;
			}
		}

		int m_Descriptor = -1;
		PosixFileIdentity m_Identity;
		FileAccess m_Access = FileAccess::ReadOnly;
		bool m_HasRegistryEntry = false;
	};
}
