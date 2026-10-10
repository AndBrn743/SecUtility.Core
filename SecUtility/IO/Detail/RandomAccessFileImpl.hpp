// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/IO/RandomAccessFile.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <limits>
#include <mutex>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#if defined(SECUTILITY_CORE_COMPILED_IMPLEMENTATION)
#define SECUTILITY_RANDOM_ACCESS_FILE_INLINE
#else
#define SECUTILITY_RANDOM_ACCESS_FILE_INLINE inline
#endif

namespace SecUtility::IO::RandomAccessFileDetail
{
	struct NativeIdentity
	{
		std::array<UInt64, 3> Components{};
		friend bool operator==(const NativeIdentity& left, const NativeIdentity& right) noexcept
		{
			return left.Components == right.Components;
		}
	};

	struct NativeIdentityHash
	{
		std::size_t operator()(const NativeIdentity& identity) const noexcept
		{
			std::size_t result = 0;
			for (const UInt64 component : identity.Components)
			{
				result ^= std::hash<UInt64>{}(component) + 0x9E3779B9U + (result << 6U) + (result >> 2U);
			}
			return result;
		}
	};

	struct RegistryEntry
	{
		std::size_t SharedCount = 0;
		bool HasExclusive = false;
	};

	struct Registry
	{
		std::mutex Mutex;
		std::unordered_map<NativeIdentity, RegistryEntry, NativeIdentityHash> Entries;
	};

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE Registry& GetRegistry()
	{
		static Registry registry;
		return registry;
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE std::string NativeError(const char* operation, const int error)
	{
#if defined(_WIN32)
		return std::string(operation) + " failed with Windows error " + std::to_string(error);
#else
		return std::string(operation) + ": " + std::generic_category().message(error);
#endif
	}

	struct FileState
	{
#if defined(_WIN32)
		HANDLE Handle = INVALID_HANDLE_VALUE;
#else
		int Descriptor = -1;
#endif
		NativeIdentity Identity;
		bool IsWritable = false;
		std::mutex LockMutex;
		bool HasLock = false;

		~FileState() noexcept
		{
#if defined(_WIN32)
			if (Handle != INVALID_HANDLE_VALUE)
			{
				::CloseHandle(Handle);
			}
#else
			if (Descriptor != -1)
			{
				::close(Descriptor);
			}
#endif
		}
	};

	struct ReadOnlyMappingState
	{
#if defined(_WIN32)
		HANDLE MappingHandle = nullptr;
#endif
		void* BasePtr = nullptr;
		std::size_t MappedBytes = 0;
		const Byte* ExposedPtr = nullptr;
		std::size_t ExposedBytes = 0;

		~ReadOnlyMappingState() noexcept
		{
#if defined(_WIN32)
			if (BasePtr != nullptr)
			{
				::UnmapViewOfFile(BasePtr);
			}
			if (MappingHandle != nullptr)
			{
				::CloseHandle(MappingHandle);
			}
#else
			if (BasePtr != nullptr)
			{
				::munmap(BasePtr, MappedBytes);
			}
#endif
		}
	};

	struct ReadWriteMappingState
	{
#if defined(_WIN32)
		HANDLE MappingHandle = nullptr;
#endif
		void* BasePtr = nullptr;
		std::size_t MappedBytes = 0;
		Byte* ExposedPtr = nullptr;
		std::size_t ExposedBytes = 0;

		~ReadWriteMappingState() noexcept
		{
#if defined(_WIN32)
			if (BasePtr != nullptr)
			{
				::UnmapViewOfFile(BasePtr);
			}
			if (MappingHandle != nullptr)
			{
				::CloseHandle(MappingHandle);
			}
#else
			if (BasePtr != nullptr)
			{
				::munmap(BasePtr, MappedBytes);
			}
#endif
		}
	};

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE void RequireState(const std::shared_ptr<FileState>& statePtr)
	{
		if (statePtr == nullptr)
		{
			throw InvalidOperationException("RandomAccessFile has no open file");
		}
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE void RequireWritable(const FileState& state)
	{
		if (!state.IsWritable)
		{
			throw InvalidOperationException("cannot perform a write operation through a read-only RandomAccessFile");
		}
	}

	template <typename ByteType>
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE void ValidateTransfer(const UInt64 offset,
	                                                           ByteType* bytesPtr,
	                                                           const std::size_t bytes)
	{
		if (bytesPtr == nullptr && bytes != 0)
		{
			throw InvalidArgumentException("exact I/O received a null buffer");
		}
#if defined(_WIN32)
		constexpr UInt64 maximumOffset = static_cast<UInt64>(std::numeric_limits<LONGLONG>::max());
#else
		constexpr UInt64 maximumOffset = static_cast<UInt64>(std::numeric_limits<off_t>::max());
#endif
		if (offset > maximumOffset || bytes > maximumOffset - offset)
		{
			throw ArgumentOutOfRangeException("exact I/O range is not representable by the platform");
		}
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE UInt64 GetSize(const FileState& state)
	{
#if defined(_WIN32)
		LARGE_INTEGER size{};
		if (!::GetFileSizeEx(state.Handle, &size))
		{
			throw IOException(NativeError("GetFileSizeEx", static_cast<int>(::GetLastError())));
		}
		if (size.QuadPart < 0)
		{
			throw IOException("file has a negative length");
		}
		return static_cast<UInt64>(size.QuadPart);
#else
		struct stat status{};
		if (::fstat(state.Descriptor, &status) != 0)
		{
			throw IOException(NativeError("fstat", errno));
		}
		if (status.st_size < 0)
		{
			throw IOException("file has a negative length");
		}
		return static_cast<UInt64>(status.st_size);
#endif
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE std::shared_ptr<FileState> AdoptNativeFile(
#if defined(_WIN32)
	        const HANDLE handle,
#else
	        const int descriptor,
#endif
	        const bool isWritable,
	        const std::filesystem::path& path)
	{
		struct NativeFileGuard
		{
#if defined(_WIN32)
			HANDLE Handle;
			~NativeFileGuard() noexcept
			{
				if (Handle != INVALID_HANDLE_VALUE)
				{
					::CloseHandle(Handle);
				}
			}
#else
			int Descriptor;
			~NativeFileGuard() noexcept
			{
				if (Descriptor != -1)
				{
					::close(Descriptor);
				}
			}
#endif
		};
#if defined(_WIN32)
		NativeFileGuard guard{handle};
#else
		NativeFileGuard guard{descriptor};
#endif
		auto statePtr = std::make_shared<FileState>();
#if defined(_WIN32)
		statePtr->Handle = handle;
		guard.Handle = INVALID_HANDLE_VALUE;
		BY_HANDLE_FILE_INFORMATION information{};
		if (!::GetFileInformationByHandle(handle, &information)
		    || (information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0)
		{
			const int error = static_cast<int>(::GetLastError());
			throw IOException("path is not an open regular file",
			                  path.string(),
			                  NativeError("GetFileInformationByHandle", error));
		}
		statePtr->Identity.Components = {
		        information.dwVolumeSerialNumber, information.nFileIndexHigh, information.nFileIndexLow};
#else
		statePtr->Descriptor = descriptor;
		guard.Descriptor = -1;
		struct stat status{};
		if (::fstat(descriptor, &status) != 0)
		{
			throw IOException("could not inspect opened file", path.string(), NativeError("fstat", errno));
		}
		if (!S_ISREG(status.st_mode))
		{
			throw IOException("path is not a regular file", path.string());
		}
		statePtr->Identity.Components = {static_cast<UInt64>(status.st_dev), static_cast<UInt64>(status.st_ino), 0};
#endif
		statePtr->IsWritable = isWritable;
		return statePtr;
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE std::shared_ptr<FileState> Open(const std::filesystem::path& path,
	                                                                     const bool isWritable,
	                                                                     const bool isCreate,
	                                                                     bool& out_alreadyExists)
	{
		out_alreadyExists = false;
#if defined(_WIN32)
		const DWORD access = isWritable ? GENERIC_READ | GENERIC_WRITE : GENERIC_READ;
		const HANDLE handle = ::CreateFileW(path.c_str(),
		                                    access,
		                                    FILE_SHARE_READ | FILE_SHARE_WRITE,
		                                    nullptr,
		                                    isCreate ? CREATE_NEW : OPEN_EXISTING,
		                                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
		                                    nullptr);
		if (handle == INVALID_HANDLE_VALUE)
		{
			const DWORD error = ::GetLastError();
			if (isCreate && (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS))
			{
				out_alreadyExists = true;
				return {};
			}
			throw IOException("could not open random-access file",
			                  path.string(),
			                  NativeError("CreateFileW", static_cast<int>(error)));
		}
		return AdoptNativeFile(handle, isWritable, path);
#else
		int flags = isWritable ? O_RDWR : O_RDONLY;
		if (isCreate)
		{
			flags |= O_CREAT | O_EXCL;
		}
#if defined(O_CLOEXEC)
		flags |= O_CLOEXEC;
#endif
		const int descriptor = ::open(path.c_str(), flags, 0666);
		if (descriptor == -1)
		{
			const int error = errno;
			if (isCreate && error == EEXIST)
			{
				out_alreadyExists = true;
				return {};
			}
			throw IOException("could not open random-access file", path.string(), NativeError("open", error));
		}
		return AdoptNativeFile(descriptor, isWritable, path);
#endif
	}

#if defined(_WIN32)
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE DWORD
	Transfer(const FileState& state, const bool isWrite, const UInt64 offset, Byte* bytesPtr, const DWORD bytes)
	{
		struct EventGuard
		{
			HANDLE Handle = nullptr;
			~EventGuard() noexcept
			{
				if (Handle != nullptr)
				{
					::CloseHandle(Handle);
				}
			}
		};
		OVERLAPPED overlapped{};
		EventGuard event{::CreateEventW(nullptr, TRUE, FALSE, nullptr)};
		if (event.Handle == nullptr)
		{
			throw IOException(NativeError("CreateEventW", static_cast<int>(::GetLastError())));
		}
		overlapped.hEvent = event.Handle;
		overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
		overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32U);
		DWORD transferred = 0;
		const BOOL started = isWrite ? ::WriteFile(state.Handle, bytesPtr, bytes, &transferred, &overlapped)
		                             : ::ReadFile(state.Handle, bytesPtr, bytes, &transferred, &overlapped);
		if (!started)
		{
			const DWORD error = ::GetLastError();
			if (error != ERROR_IO_PENDING || !::GetOverlappedResult(state.Handle, &overlapped, &transferred, TRUE))
			{
				const DWORD finalError = error == ERROR_IO_PENDING ? ::GetLastError() : error;
				throw IOException(NativeError(isWrite ? "WriteFile" : "ReadFile", static_cast<int>(finalError)));
			}
		}
		return transferred;
	}
#endif

	template <typename ByteType>
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE void ExactTransfer(const FileState& state,
	                                                        const bool isWrite,
	                                                        const UInt64 offset,
	                                                        ByteType* bytesPtr,
	                                                        const std::size_t bytes)
	{
		std::size_t completed = 0;
		while (completed < bytes)
		{
			std::size_t transferred = 0;
#if defined(_WIN32)
			const DWORD chunk = static_cast<DWORD>(std::min<std::size_t>(bytes - completed, MAXDWORD));
			transferred = Transfer(state, isWrite, offset + completed, const_cast<Byte*>(bytesPtr + completed), chunk);
#else
			const std::size_t chunk = std::min<std::size_t>(
			        bytes - completed, static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
			const ssize_t result = isWrite ? ::pwrite(state.Descriptor,
			                                          bytesPtr + completed,
			                                          chunk,
			                                          static_cast<off_t>(offset + completed))
			                               : ::pread(state.Descriptor,
			                                         const_cast<Byte*>(bytesPtr + completed),
			                                         chunk,
			                                         static_cast<off_t>(offset + completed));
			if (result < 0)
			{
				if (errno == EINTR)
				{
					continue;
				}
				throw IOException(NativeError(isWrite ? "pwrite" : "pread", errno));
			}
			transferred = static_cast<std::size_t>(result);
#endif
			if (transferred == 0)
			{
				throw IOException(isWrite ? "zero-length result during exact write"
				                          : "unexpected end of file during exact read");
			}
			completed += transferred;
		}
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE UInt64 MappingGranularity()
	{
#if defined(_WIN32)
		SYSTEM_INFO information{};
		::GetSystemInfo(&information);
		if (information.dwAllocationGranularity == 0)
		{
			throw IOException("could not query mapping granularity");
		}
		return information.dwAllocationGranularity;
#else
		const long bytes = ::sysconf(_SC_PAGE_SIZE);
		if (bytes <= 0)
		{
			throw IOException("could not query mapping granularity");
		}
		return static_cast<UInt64>(bytes);
#endif
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE std::pair<UInt64, std::size_t> ValidateMapping(const FileState& state,
	                                                                                    const UInt64 offset,
	                                                                                    const std::size_t bytes)
	{
#if defined(_WIN32)
		constexpr UInt64 maximumOffset = static_cast<UInt64>(std::numeric_limits<LONGLONG>::max());
#else
		constexpr UInt64 maximumOffset = static_cast<UInt64>(std::numeric_limits<off_t>::max());
#endif
		if (offset > maximumOffset || bytes > maximumOffset - offset)
		{
			throw ArgumentOutOfRangeException("mapping range is not representable by the platform");
		}
		const UInt64 fileBytes = GetSize(state);
		if (offset > fileBytes || bytes > fileBytes - offset)
		{
			throw ArgumentOutOfRangeException("mapping extends past the file length");
		}
		const UInt64 granularity = MappingGranularity();
		const UInt64 mappedOffset = offset - offset % granularity;
		const UInt64 delta = offset - mappedOffset;
		if (bytes > std::numeric_limits<std::size_t>::max() - delta)
		{
			throw ArgumentOutOfRangeException("native mapping size is not representable");
		}
		return {mappedOffset, static_cast<std::size_t>(delta) + bytes};
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE bool AcquireRegistry(const NativeIdentity& identity,
	                                                          const RandomAccessFile::LockMode mode)
	{
		Registry& registry = GetRegistry();
		std::lock_guard<std::mutex> guard(registry.Mutex);
		auto& entry = registry.Entries[identity];
		const bool hasConflict = mode == RandomAccessFile::LockMode::Shared
		                                 ? entry.HasExclusive
		                                 : entry.HasExclusive || entry.SharedCount != 0;
		if (hasConflict)
		{
			if (entry.SharedCount == 0 && !entry.HasExclusive)
			{
				registry.Entries.erase(identity);
			}
			return false;
		}
		if (mode == RandomAccessFile::LockMode::Shared)
		{
			++entry.SharedCount;
		}
		else
		{
			entry.HasExclusive = true;
		}
		return true;
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE void ReleaseRegistry(const NativeIdentity& identity,
	                                                          const RandomAccessFile::LockMode mode) noexcept
	{
		Registry& registry = GetRegistry();
		std::lock_guard<std::mutex> guard(registry.Mutex);
		const auto iterator = registry.Entries.find(identity);
		if (iterator == registry.Entries.end())
		{
			return;
		}
		if (mode == RandomAccessFile::LockMode::Shared)
		{
			if (iterator->second.SharedCount != 0)
			{
				--iterator->second.SharedCount;
			}
		}
		else
		{
			iterator->second.HasExclusive = false;
		}
		if (iterator->second.SharedCount == 0 && !iterator->second.HasExclusive)
		{
			registry.Entries.erase(iterator);
		}
	}
}

namespace SecUtility::IO
{
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE ReadOnlyMappedRegion::ReadOnlyMappedRegion(
	        std::unique_ptr<RandomAccessFileDetail::ReadOnlyMappingState> statePtr) noexcept
	    : m_StatePtr(std::move(statePtr))
	{
	}
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE ReadOnlyMappedRegion::ReadOnlyMappedRegion(ReadOnlyMappedRegion&&) noexcept =
	        default;
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE ReadOnlyMappedRegion& ReadOnlyMappedRegion::operator=(
	        ReadOnlyMappedRegion&&) noexcept = default;
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE ReadOnlyMappedRegion::~ReadOnlyMappedRegion() noexcept = default;
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE const Byte* ReadOnlyMappedRegion::Data() const noexcept
	{
		return m_StatePtr->ExposedPtr;
	}
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE std::size_t ReadOnlyMappedRegion::Size() const noexcept
	{
		return m_StatePtr->ExposedBytes;
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE ReadWriteMappedRegion::ReadWriteMappedRegion(
	        std::unique_ptr<RandomAccessFileDetail::ReadWriteMappingState> statePtr) noexcept
	    : m_StatePtr(std::move(statePtr))
	{
	}
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE ReadWriteMappedRegion::ReadWriteMappedRegion(
	        ReadWriteMappedRegion&&) noexcept = default;
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE ReadWriteMappedRegion& ReadWriteMappedRegion::operator=(
	        ReadWriteMappedRegion&&) noexcept = default;
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE ReadWriteMappedRegion::~ReadWriteMappedRegion() noexcept = default;
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE Byte* ReadWriteMappedRegion::Data() const noexcept
	{
		return m_StatePtr->ExposedPtr;
	}
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE std::size_t ReadWriteMappedRegion::Size() const noexcept
	{
		return m_StatePtr->ExposedBytes;
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE void ReadWriteMappedRegion::Complete()
	{
		if (m_StatePtr->BasePtr == nullptr)
		{
			return;
		}
#if defined(_WIN32)
		if (!::FlushViewOfFile(m_StatePtr->BasePtr, 0))
		{
			throw IOException(
			        RandomAccessFileDetail::NativeError("FlushViewOfFile", static_cast<int>(::GetLastError())));
		}
		if (!::UnmapViewOfFile(m_StatePtr->BasePtr))
		{
			throw IOException(
			        RandomAccessFileDetail::NativeError("UnmapViewOfFile", static_cast<int>(::GetLastError())));
		}
		m_StatePtr->BasePtr = nullptr;
		m_StatePtr->ExposedPtr = nullptr;
		m_StatePtr->ExposedBytes = 0;
		const HANDLE mappingHandle = std::exchange(m_StatePtr->MappingHandle, nullptr);
		if (!::CloseHandle(mappingHandle))
		{
			throw IOException(RandomAccessFileDetail::NativeError("CloseHandle", static_cast<int>(::GetLastError())));
		}
#else
		if (::msync(m_StatePtr->BasePtr, m_StatePtr->MappedBytes, MS_SYNC) != 0)
		{
			throw IOException(RandomAccessFileDetail::NativeError("msync", errno));
		}
		if (::munmap(m_StatePtr->BasePtr, m_StatePtr->MappedBytes) != 0)
		{
			throw IOException(RandomAccessFileDetail::NativeError("munmap", errno));
		}
		m_StatePtr->BasePtr = nullptr;
		m_StatePtr->ExposedPtr = nullptr;
		m_StatePtr->ExposedBytes = 0;
		m_StatePtr->MappedBytes = 0;
#endif
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE RandomAccessFile
	RandomAccessFile::OpenForReadOnly(const std::filesystem::path& path)
	{
		bool ignored = false;
		return RandomAccessFile(RandomAccessFileDetail::Open(path, false, false, ignored));
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE RandomAccessFile
	RandomAccessFile::OpenForReadWrite(const std::filesystem::path& path)
	{
		bool ignored = false;
		return RandomAccessFile(RandomAccessFileDetail::Open(path, true, false, ignored));
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE std::optional<RandomAccessFile> RandomAccessFile::TryCreateExclusive(
	        const std::filesystem::path& path)
	{
		bool alreadyExists = false;
		auto statePtr = RandomAccessFileDetail::Open(path, true, true, alreadyExists);
		if (alreadyExists)
		{
			return std::nullopt;
		}
		return RandomAccessFile(std::move(statePtr));
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE UInt64 RandomAccessFile::Size() const
	{
		RandomAccessFileDetail::RequireState(m_StatePtr);
		return RandomAccessFileDetail::GetSize(*m_StatePtr);
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE void RandomAccessFile::Resize(const UInt64 bytes)
	{
		RandomAccessFileDetail::RequireState(m_StatePtr);
		RandomAccessFileDetail::RequireWritable(*m_StatePtr);
#if defined(_WIN32)
		if (bytes > static_cast<UInt64>(std::numeric_limits<LONGLONG>::max()))
		{
			throw ArgumentOutOfRangeException("file size exceeds the Windows offset range");
		}
		LARGE_INTEGER position{};
		position.QuadPart = static_cast<LONGLONG>(bytes);
		if (!::SetFilePointerEx(m_StatePtr->Handle, position, nullptr, FILE_BEGIN)
		    || !::SetEndOfFile(m_StatePtr->Handle))
		{
			throw IOException(RandomAccessFileDetail::NativeError("SetEndOfFile", static_cast<int>(::GetLastError())));
		}
#else
		if (bytes > static_cast<UInt64>(std::numeric_limits<off_t>::max()))
		{
			throw ArgumentOutOfRangeException("file size exceeds the POSIX offset range");
		}
		if (::ftruncate(m_StatePtr->Descriptor, static_cast<off_t>(bytes)) != 0)
		{
			throw IOException(RandomAccessFileDetail::NativeError("ftruncate", errno));
		}
#endif
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE void RandomAccessFile::ReadExactAt(const UInt64 offset,
	                                                                        Byte* out_bytesPtr,
	                                                                        const std::size_t bytes) const
	{
		RandomAccessFileDetail::RequireState(m_StatePtr);
		RandomAccessFileDetail::ValidateTransfer(offset, out_bytesPtr, bytes);
		RandomAccessFileDetail::ExactTransfer(*m_StatePtr, false, offset, out_bytesPtr, bytes);
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE void RandomAccessFile::WriteExactAt(const UInt64 offset,
	                                                                         const Byte* bytesPtr,
	                                                                         const std::size_t bytes)
	{
		RandomAccessFileDetail::RequireState(m_StatePtr);
		RandomAccessFileDetail::RequireWritable(*m_StatePtr);
		RandomAccessFileDetail::ValidateTransfer(offset, bytesPtr, bytes);
		RandomAccessFileDetail::ExactTransfer(*m_StatePtr, true, offset, bytesPtr, bytes);
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE ReadOnlyMappedRegion
	RandomAccessFile::MapReadOnly(const UInt64 offset, const std::size_t bytes) const
	{
		RandomAccessFileDetail::RequireState(m_StatePtr);
		auto statePtr = std::make_unique<RandomAccessFileDetail::ReadOnlyMappingState>();
		statePtr->ExposedBytes = bytes;
		const auto [mappedOffset, mappedBytes] = RandomAccessFileDetail::ValidateMapping(*m_StatePtr, offset, bytes);
		if (bytes == 0)
		{
			return ReadOnlyMappedRegion(std::move(statePtr));
		}
		const std::size_t delta = static_cast<std::size_t>(offset - mappedOffset);
		statePtr->MappedBytes = mappedBytes;
#if defined(_WIN32)
		statePtr->MappingHandle = ::CreateFileMappingW(m_StatePtr->Handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
		if (statePtr->MappingHandle == nullptr)
		{
			throw IOException(
			        RandomAccessFileDetail::NativeError("CreateFileMappingW", static_cast<int>(::GetLastError())));
		}
		statePtr->BasePtr = ::MapViewOfFile(statePtr->MappingHandle,
		                                    FILE_MAP_READ,
		                                    static_cast<DWORD>(mappedOffset >> 32U),
		                                    static_cast<DWORD>(mappedOffset & 0xFFFFFFFFULL),
		                                    mappedBytes);
		if (statePtr->BasePtr == nullptr)
		{
			throw IOException(RandomAccessFileDetail::NativeError("MapViewOfFile", static_cast<int>(::GetLastError())));
		}
#else
		statePtr->BasePtr = ::mmap(
		        nullptr, mappedBytes, PROT_READ, MAP_SHARED, m_StatePtr->Descriptor, static_cast<off_t>(mappedOffset));
		if (statePtr->BasePtr == MAP_FAILED)
		{
			statePtr->BasePtr = nullptr;
			throw IOException(RandomAccessFileDetail::NativeError("mmap", errno));
		}
#endif
		statePtr->ExposedPtr = static_cast<const Byte*>(statePtr->BasePtr) + delta;
		return ReadOnlyMappedRegion(std::move(statePtr));
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE ReadWriteMappedRegion RandomAccessFile::MapReadWrite(const UInt64 offset,
	                                                                                          const std::size_t bytes)
	{
		RandomAccessFileDetail::RequireState(m_StatePtr);
		RandomAccessFileDetail::RequireWritable(*m_StatePtr);
		auto statePtr = std::make_unique<RandomAccessFileDetail::ReadWriteMappingState>();
		statePtr->ExposedBytes = bytes;
		const auto [mappedOffset, mappedBytes] = RandomAccessFileDetail::ValidateMapping(*m_StatePtr, offset, bytes);
		if (bytes == 0)
		{
			return ReadWriteMappedRegion(std::move(statePtr));
		}
		const std::size_t delta = static_cast<std::size_t>(offset - mappedOffset);
		statePtr->MappedBytes = mappedBytes;
#if defined(_WIN32)
		statePtr->MappingHandle = ::CreateFileMappingW(m_StatePtr->Handle, nullptr, PAGE_READWRITE, 0, 0, nullptr);
		if (statePtr->MappingHandle == nullptr)
		{
			throw IOException(
			        RandomAccessFileDetail::NativeError("CreateFileMappingW", static_cast<int>(::GetLastError())));
		}
		statePtr->BasePtr = ::MapViewOfFile(statePtr->MappingHandle,
		                                    FILE_MAP_READ | FILE_MAP_WRITE,
		                                    static_cast<DWORD>(mappedOffset >> 32U),
		                                    static_cast<DWORD>(mappedOffset & 0xFFFFFFFFULL),
		                                    mappedBytes);
		if (statePtr->BasePtr == nullptr)
		{
			throw IOException(RandomAccessFileDetail::NativeError("MapViewOfFile", static_cast<int>(::GetLastError())));
		}
#else
		statePtr->BasePtr = ::mmap(nullptr,
		                           mappedBytes,
		                           PROT_READ | PROT_WRITE,
		                           MAP_SHARED,
		                           m_StatePtr->Descriptor,
		                           static_cast<off_t>(mappedOffset));
		if (statePtr->BasePtr == MAP_FAILED)
		{
			statePtr->BasePtr = nullptr;
			throw IOException(RandomAccessFileDetail::NativeError("mmap", errno));
		}
#endif
		statePtr->ExposedPtr = static_cast<Byte*>(statePtr->BasePtr) + delta;
		return ReadWriteMappedRegion(std::move(statePtr));
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE FileIdentity RandomAccessFile::GetIdentity() const
	{
		RandomAccessFileDetail::RequireState(m_StatePtr);
		return FileIdentity(m_StatePtr->Identity.Components);
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE bool RandomAccessFile::IdentifiesSameFile(
	        const std::filesystem::path& path) const noexcept
	{
		if (m_StatePtr == nullptr)
		{
			return false;
		}
#if defined(_WIN32)
		const HANDLE handle = ::CreateFileW(path.c_str(),
		                                    FILE_READ_ATTRIBUTES,
		                                    FILE_SHARE_READ | FILE_SHARE_WRITE,
		                                    nullptr,
		                                    OPEN_EXISTING,
		                                    FILE_ATTRIBUTE_NORMAL,
		                                    nullptr);
		if (handle == INVALID_HANDLE_VALUE)
		{
			return false;
		}
		BY_HANDLE_FILE_INFORMATION information{};
		const bool isSame = ::GetFileInformationByHandle(handle, &information)
		                    && RandomAccessFileDetail::NativeIdentity{{information.dwVolumeSerialNumber,
		                                                               information.nFileIndexHigh,
		                                                               information.nFileIndexLow}}
		                               == m_StatePtr->Identity;
		::CloseHandle(handle);
		return isSame;
#else
		struct stat status{};
		return ::stat(path.c_str(), &status) == 0
		       && RandomAccessFileDetail::NativeIdentity{{static_cast<UInt64>(status.st_dev),
		                                                  static_cast<UInt64>(status.st_ino),
		                                                  0}}
		                  == m_StatePtr->Identity;
#endif
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE std::optional<RandomAccessFile::Lock> RandomAccessFile::TryAcquireLock(
	        const LockMode mode) const
	{
		RandomAccessFileDetail::RequireState(m_StatePtr);
		if (mode == LockMode::Exclusive)
		{
			RandomAccessFileDetail::RequireWritable(*m_StatePtr);
		}
		std::lock_guard<std::mutex> stateGuard(m_StatePtr->LockMutex);
		if (m_StatePtr->HasLock)
		{
			return std::nullopt;
		}
		if (!RandomAccessFileDetail::AcquireRegistry(m_StatePtr->Identity, mode))
		{
			return std::nullopt;
		}
#if defined(_WIN32)
		OVERLAPPED overlapped{};
		DWORD flags = LOCKFILE_FAIL_IMMEDIATELY;
		if (mode == LockMode::Exclusive)
		{
			flags |= LOCKFILE_EXCLUSIVE_LOCK;
		}
		if (!::LockFileEx(m_StatePtr->Handle, flags, 0, MAXDWORD, MAXDWORD, &overlapped))
		{
			const DWORD error = ::GetLastError();
			RandomAccessFileDetail::ReleaseRegistry(m_StatePtr->Identity, mode);
			if (error == ERROR_LOCK_VIOLATION)
			{
				return std::nullopt;
			}
			throw IOException(RandomAccessFileDetail::NativeError("LockFileEx", static_cast<int>(error)));
		}
#else
#if defined(F_OFD_SETLK) && defined(__linux__)
		struct flock lock{};
		lock.l_type = mode == LockMode::Shared ? F_RDLCK : F_WRLCK;
		lock.l_whence = SEEK_SET;
		lock.l_len = 0;
		const int result = ::fcntl(m_StatePtr->Descriptor, F_OFD_SETLK, &lock);
#else
		const int operation = (mode == LockMode::Shared ? LOCK_SH : LOCK_EX) | LOCK_NB;
		const int result = ::flock(m_StatePtr->Descriptor, operation);
#endif
		if (result != 0)
		{
			const int error = errno;
			RandomAccessFileDetail::ReleaseRegistry(m_StatePtr->Identity, mode);
			if (error == EACCES || error == EAGAIN)
			{
				return std::nullopt;
			}
			throw IOException(RandomAccessFileDetail::NativeError("file lock", error));
		}
#endif
		m_StatePtr->HasLock = true;
		return Lock(m_StatePtr, mode);
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE RandomAccessFile::Lock::Lock(
	        std::shared_ptr<RandomAccessFileDetail::FileState> statePtr, const LockMode mode) noexcept
	    : m_StatePtr(std::move(statePtr)), m_Mode(mode)
	{
	}
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE RandomAccessFile::Lock::Lock(Lock&& other) noexcept
	    : m_StatePtr(std::move(other.m_StatePtr)), m_Mode(other.m_Mode)
	{
	}
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE RandomAccessFile::Lock& RandomAccessFile::Lock::operator=(
	        Lock&& other) noexcept
	{
		if (this != &other)
		{
			Reset();
			m_StatePtr = std::move(other.m_StatePtr);
			m_Mode = other.m_Mode;
		}
		return *this;
	}
	SECUTILITY_RANDOM_ACCESS_FILE_INLINE RandomAccessFile::Lock::~Lock() noexcept
	{
		Reset();
	}

	SECUTILITY_RANDOM_ACCESS_FILE_INLINE void RandomAccessFile::Lock::Reset() noexcept
	{
		if (m_StatePtr == nullptr)
		{
			return;
		}
		auto statePtr = std::move(m_StatePtr);
		std::lock_guard<std::mutex> stateGuard(statePtr->LockMutex);
#if defined(_WIN32)
		OVERLAPPED overlapped{};
		::UnlockFileEx(statePtr->Handle, 0, MAXDWORD, MAXDWORD, &overlapped);
#else
#if defined(F_OFD_SETLK) && defined(__linux__)
		struct flock lock{};
		lock.l_type = F_UNLCK;
		lock.l_whence = SEEK_SET;
		lock.l_len = 0;
		::fcntl(statePtr->Descriptor, F_OFD_SETLK, &lock);
#else
		::flock(statePtr->Descriptor, LOCK_UN);
#endif
#endif
		RandomAccessFileDetail::ReleaseRegistry(statePtr->Identity, m_Mode);
		statePtr->HasLock = false;
	}
}

#undef SECUTILITY_RANDOM_ACCESS_FILE_INLINE
