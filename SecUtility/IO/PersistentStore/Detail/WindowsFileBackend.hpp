// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#if !defined(_WIN32)
#error WindowsFileBackend.hpp can only be used on Windows
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/IO/PersistentStore/Detail/CheckedArithmetic.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Format.hpp>
#include <SecUtility/Raw/Int.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>


namespace SecUtility::IO::PersistentStoreDetail
{
	template <bool IsWritable>
	class WindowsMappedRegion final
	{
	public:
		WindowsMappedRegion() noexcept = default;

		WindowsMappedRegion(const HANDLE mappingHandle,
		                    void* const basePtr,
		                    Byte* const exposedPtr,
		                    const std::size_t exposedBytes) noexcept
		    : m_MappingHandle(mappingHandle), m_BasePtr(basePtr), m_ExposedPtr(exposedPtr),
		      m_ExposedBytes(exposedBytes)
		{
			/* NO CODE */
		}

		WindowsMappedRegion(WindowsMappedRegion&& other) noexcept
		    : m_MappingHandle(std::exchange(other.m_MappingHandle, nullptr)),
		      m_BasePtr(std::exchange(other.m_BasePtr, nullptr)),
		      m_ExposedPtr(std::exchange(other.m_ExposedPtr, nullptr)),
		      m_ExposedBytes(std::exchange(other.m_ExposedBytes, 0))
		{
			/* NO CODE */
		}

		WindowsMappedRegion& operator=(WindowsMappedRegion&& other) noexcept
		{
			if (this != &other)
			{
				Reset();
				m_MappingHandle = std::exchange(other.m_MappingHandle, nullptr);
				m_BasePtr = std::exchange(other.m_BasePtr, nullptr);
				m_ExposedPtr = std::exchange(other.m_ExposedPtr, nullptr);
				m_ExposedBytes = std::exchange(other.m_ExposedBytes, 0);
			}
			return *this;
		}

		WindowsMappedRegion(const WindowsMappedRegion&) = delete;
		WindowsMappedRegion& operator=(const WindowsMappedRegion&) = delete;
		~WindowsMappedRegion() noexcept { Reset(); }

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

		std::size_t Size() const noexcept { return m_ExposedBytes; }

		template <bool IsEnabled = IsWritable, std::enable_if_t<IsEnabled, int> = 0>
		void Complete()
		{
			if (m_BasePtr == nullptr)
			{
				return;
			}
			if (!::FlushViewOfFile(m_BasePtr, 0))
			{
				throw IOException("FlushViewOfFile failed with Windows error " + std::to_string(::GetLastError()));
			}
			if (!::UnmapViewOfFile(m_BasePtr))
			{
				throw IOException("UnmapViewOfFile failed with Windows error " + std::to_string(::GetLastError()));
			}
			m_BasePtr = nullptr;
			m_ExposedPtr = nullptr;
			m_ExposedBytes = 0;
			const HANDLE mappingHandle = std::exchange(m_MappingHandle, nullptr);
			if (!::CloseHandle(mappingHandle))
			{
				throw IOException("CloseHandle failed with Windows error " + std::to_string(::GetLastError()));
			}
		}

	private:
		void Reset() noexcept
		{
			if (m_BasePtr != nullptr)
			{
				::UnmapViewOfFile(m_BasePtr);
			}
			if (m_MappingHandle != nullptr)
			{
				::CloseHandle(m_MappingHandle);
			}
			m_MappingHandle = nullptr;
			m_BasePtr = nullptr;
			m_ExposedPtr = nullptr;
			m_ExposedBytes = 0;
		}

		HANDLE m_MappingHandle = nullptr;
		void* m_BasePtr = nullptr;
		Byte* m_ExposedPtr = nullptr;
		std::size_t m_ExposedBytes = 0;
	};

	using WindowsReadOnlyMappedRegion = WindowsMappedRegion<false>;
	using WindowsWritableMappedRegion = WindowsMappedRegion<true>;

	struct WindowsFileIdentity
	{
		DWORD VolumeSerialNumber = 0;
		DWORD FileIndexHigh = 0;
		DWORD FileIndexLow = 0;

		friend bool operator==(const WindowsFileIdentity& left, const WindowsFileIdentity& right) noexcept
		{
			return left.VolumeSerialNumber == right.VolumeSerialNumber && left.FileIndexHigh == right.FileIndexHigh
			       && left.FileIndexLow == right.FileIndexLow;
		}
	};

	struct WindowsFileIdentityHash
	{
		std::size_t operator()(const WindowsFileIdentity& identity) const noexcept
		{
			std::size_t result = std::hash<DWORD>{}(identity.VolumeSerialNumber);
			result ^= std::hash<DWORD>{}(identity.FileIndexHigh) + 0x9E3779B9U + (result << 6U) + (result >> 2U);
			result ^= std::hash<DWORD>{}(identity.FileIndexLow) + 0x9E3779B9U + (result << 6U) + (result >> 2U);
			return result;
		}
	};

	namespace WindowsFileBackendDetail
	{
		struct RegistryEntry
		{
			std::size_t ReaderCount = 0;
			bool HasWriter = false;
		};

		inline std::mutex RegistryMutex;
		inline std::unordered_map<WindowsFileIdentity, RegistryEntry, WindowsFileIdentityHash> Registry;

		inline std::string ErrorMessage(const char* operation, const DWORD error)
		{
			return std::string(operation) + " failed with Windows error " + std::to_string(error);
		}

		inline void AcquireRegistry(const WindowsFileIdentity& identity, const FileAccess access)
		{
			std::lock_guard<std::mutex> lock(RegistryMutex);
			auto iterator = Registry.find(identity);
			if (iterator == Registry.end())
			{
				iterator = Registry.emplace(identity, RegistryEntry{}).first;
			}
			RegistryEntry& entry = iterator->second;
			const bool hasConflict = access == FileAccess::ReadOnly ? entry.HasWriter
			                                                        : entry.HasWriter || entry.ReaderCount != 0;
			if (hasConflict)
			{
				throw IOException("PersistentStore lock unavailable in this process");
			}
			if (access == FileAccess::ReadOnly)
			{
				++entry.ReaderCount;
			}
			else
			{
				entry.HasWriter = true;
			}
		}

		inline void ReleaseRegistry(const WindowsFileIdentity& identity, const FileAccess access) noexcept
		{
			std::lock_guard<std::mutex> lock(RegistryMutex);
			const auto iterator = Registry.find(identity);
			if (iterator == Registry.end())
			{
				return;
			}
			if (access == FileAccess::ReadOnly)
			{
				if (iterator->second.ReaderCount != 0)
				{
					--iterator->second.ReaderCount;
				}
			}
			else
			{
				iterator->second.HasWriter = false;
			}
			if (iterator->second.ReaderCount == 0 && !iterator->second.HasWriter)
			{
				Registry.erase(iterator);
			}
		}
	}

	class WindowsFileBackend final
	{
	public:
		static WindowsFileBackend Open(const std::filesystem::path& path, const FileAccess access)
		{
			const DWORD desiredAccess = access == FileAccess::ReadOnly ? GENERIC_READ : GENERIC_READ | GENERIC_WRITE;
			const HANDLE handle = ::CreateFileW(path.c_str(), desiredAccess, FILE_SHARE_READ | FILE_SHARE_WRITE,
			                                    nullptr, OPEN_EXISTING,
			                                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
			if (handle == INVALID_HANDLE_VALUE)
			{
				throw IOException("PersistentStore open failed",
				                  WindowsFileBackendDetail::ErrorMessage("CreateFileW", ::GetLastError()));
			}
			return AdoptAndLock(handle, access);
		}

		static std::optional<WindowsFileBackend> TryCreateExclusive(const std::filesystem::path& path)
		{
			const HANDLE handle = ::CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
			                                    FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, CREATE_NEW,
			                                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
			if (handle == INVALID_HANDLE_VALUE)
			{
				const DWORD error = ::GetLastError();
				if (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS)
				{
					return std::nullopt;
				}
				throw IOException("PersistentStore exclusive create failed",
				                  WindowsFileBackendDetail::ErrorMessage("CreateFileW", error));
			}
			return AdoptAndLock(handle, FileAccess::ReadWrite);
		}

		WindowsFileBackend(WindowsFileBackend&& other) noexcept
		    : m_Handle(std::exchange(other.m_Handle, INVALID_HANDLE_VALUE)),
		      m_Identity(other.m_Identity),
		      m_Access(other.m_Access),
		      m_HasRegistryEntry(std::exchange(other.m_HasRegistryEntry, false))
		{
			/* NO CODE */
		}

		WindowsFileBackend& operator=(WindowsFileBackend&& other) noexcept
		{
			if (this != &other)
			{
				Close();
				m_Handle = std::exchange(other.m_Handle, INVALID_HANDLE_VALUE);
				m_Identity = other.m_Identity;
				m_Access = other.m_Access;
				m_HasRegistryEntry = std::exchange(other.m_HasRegistryEntry, false);
			}
			return *this;
		}

		WindowsFileBackend(const WindowsFileBackend&) = delete;
		WindowsFileBackend& operator=(const WindowsFileBackend&) = delete;
		~WindowsFileBackend() noexcept { Close(); }

		UInt64 GetPhysicalFileBytes() const
		{
			LARGE_INTEGER size{};
			if (!::GetFileSizeEx(m_Handle, &size) || size.QuadPart < 0)
			{
				throw IOException(WindowsFileBackendDetail::ErrorMessage("GetFileSizeEx", ::GetLastError()));
			}
			return static_cast<UInt64>(size.QuadPart);
		}

		void SetPhysicalFileBytes(const UInt64 byteCount)
		{
			RequireWritable("grow file");
			if (byteCount > static_cast<UInt64>(std::numeric_limits<LONGLONG>::max()))
			{
				throw IOException("PersistentStore file length exceeds the Windows offset range", byteCount);
			}
			LARGE_INTEGER position{};
			position.QuadPart = static_cast<LONGLONG>(byteCount);
			if (!::SetFilePointerEx(m_Handle, position, nullptr, FILE_BEGIN) || !::SetEndOfFile(m_Handle))
			{
				throw IOException(WindowsFileBackendDetail::ErrorMessage("SetEndOfFile", ::GetLastError()));
			}
		}

		void ReadExact(const UInt64 offset, Byte* out_bytesPtr, const std::size_t byteCount) const
		{
			ValidateBufferAndRange(offset, out_bytesPtr, byteCount, "read");
			CompleteExactTransfer(offset, out_bytesPtr, byteCount, true,
			                      [this](const UInt64 currentOffset, Byte* currentBytesPtr,
			                             const std::size_t remainingBytes) {
				                      const DWORD chunk = static_cast<DWORD>(
				                              std::min<std::size_t>(remainingBytes, MAXDWORD));
				                      return ExactTransferResult{
				                              Transfer(false, currentOffset, currentBytesPtr, chunk), false};
			                      });
		}

		void WriteExact(const UInt64 offset, const Byte* bytesPtr, const std::size_t byteCount)
		{
			RequireWritable("write file");
			ValidateBufferAndRange(offset, bytesPtr, byteCount, "write");
			CompleteExactTransfer(offset, bytesPtr, byteCount, false,
			                      [this](const UInt64 currentOffset, const Byte* currentBytesPtr,
			                             const std::size_t remainingBytes) {
				                      const DWORD chunk = static_cast<DWORD>(
				                              std::min<std::size_t>(remainingBytes, MAXDWORD));
				                      return ExactTransferResult{
				                              Transfer(true, currentOffset, const_cast<Byte*>(currentBytesPtr), chunk),
				                              false};
			                      });
		}

		FileAccess GetAccess() const noexcept { return m_Access; }

		bool DoesPathIdentifySameFile(const std::filesystem::path& path) const noexcept
		{
			const HANDLE pathHandle = ::CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
			                                        FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
			                                        FILE_ATTRIBUTE_NORMAL, nullptr);
			if (pathHandle == INVALID_HANDLE_VALUE)
			{
				return false;
			}
			BY_HANDLE_FILE_INFORMATION information{};
			const bool isSame = ::GetFileInformationByHandle(pathHandle, &information)
			                    && WindowsFileIdentity{information.dwVolumeSerialNumber,
			                                           information.nFileIndexHigh,
			                                           information.nFileIndexLow}
			                               == m_Identity;
			::CloseHandle(pathHandle);
			return isSame;
		}

		static UInt64 GetMappingGranularity()
		{
			SYSTEM_INFO information{};
			::GetSystemInfo(&information);
			if (information.dwAllocationGranularity == 0)
			{
				throw IOException("PersistentStore could not query the Windows allocation granularity");
			}
			return information.dwAllocationGranularity;
		}

		WindowsReadOnlyMappedRegion MapReadOnly(const UInt64 offset,
		                                               const std::size_t byteCount,
		                                               const UInt64 alignment) const
		{
			return Map<false>(offset, byteCount, alignment);
		}

		WindowsWritableMappedRegion MapWritable(const UInt64 offset,
		                                              const std::size_t byteCount,
		                                              const UInt64 alignment)
		{
			RequireWritable("map writable file region");
			return Map<true>(offset, byteCount, alignment);
		}

	private:
		template <bool IsWritable>
		WindowsMappedRegion<IsWritable> Map(const UInt64 offset,
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
			const std::size_t mappedBytes = CheckedNarrow<std::size_t>(
			        mappedBytes64, "PersistentStore native mapped byte count");
			const DWORD protection = IsWritable ? PAGE_READWRITE : PAGE_READONLY;
			const HANDLE mappingHandle = ::CreateFileMappingW(m_Handle, nullptr, protection, 0, 0, nullptr);
			if (mappingHandle == nullptr)
			{
				throw IOException(WindowsFileBackendDetail::ErrorMessage("CreateFileMappingW", ::GetLastError()));
			}
			const DWORD access = IsWritable ? FILE_MAP_WRITE | FILE_MAP_READ : FILE_MAP_READ;
			void* const basePtr = ::MapViewOfFile(mappingHandle, access,
			                                      static_cast<DWORD>(mappedOffset >> 32U),
			                                      static_cast<DWORD>(mappedOffset & 0xFFFFFFFFULL), mappedBytes);
			if (basePtr == nullptr)
			{
				const DWORD error = ::GetLastError();
				::CloseHandle(mappingHandle);
				throw IOException(WindowsFileBackendDetail::ErrorMessage("MapViewOfFile", error));
			}
			Byte* const exposedPtr = static_cast<Byte*>(basePtr) + static_cast<std::size_t>(delta);
			if (reinterpret_cast<std::uintptr_t>(exposedPtr) % alignment != 0)
			{
				::UnmapViewOfFile(basePtr);
				::CloseHandle(mappingHandle);
				throw IOException("PersistentStore mapped address does not satisfy the requested alignment");
			}
			return {mappingHandle, basePtr, exposedPtr, byteCount};
		}

		void ValidateMappingRequest(const UInt64 offset,
		                            const std::size_t byteCount,
		                            const UInt64 alignment) const
		{
			if (alignment == 0 || alignment > MaximumPayloadAlignment || (alignment & (alignment - 1)) != 0)
			{
				throw IOException("PersistentStore mapping alignment must be a power of two from 1 through 4096");
			}
			if (offset % alignment != 0)
			{
				throw IOException("PersistentStore mapping offset does not satisfy the requested alignment");
			}
			if (offset > static_cast<UInt64>(std::numeric_limits<LONGLONG>::max())
			    || byteCount > static_cast<UInt64>(std::numeric_limits<LONGLONG>::max()) - offset)
			{
				throw IOException("PersistentStore mapping exceeds the Windows offset range");
			}
			const UInt64 physicalFileBytes = GetPhysicalFileBytes();
			if (offset > physicalFileBytes || byteCount > physicalFileBytes - offset)
			{
				throw IOException("PersistentStore mapping extends past the physical file length");
			}
		}

		WindowsFileBackend(const HANDLE handle,
		                   const WindowsFileIdentity identity,
		                   const FileAccess access) noexcept
		    : m_Handle(handle), m_Identity(identity), m_Access(access), m_HasRegistryEntry(true)
		{
			/* NO CODE */
		}

		static WindowsFileBackend AdoptAndLock(const HANDLE handle, const FileAccess access)
		{
			BY_HANDLE_FILE_INFORMATION information{};
			if (!::GetFileInformationByHandle(handle, &information)
			    || (information.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0)
			{
				const DWORD error = ::GetLastError();
				::CloseHandle(handle);
				throw IOException(WindowsFileBackendDetail::ErrorMessage("GetFileInformationByHandle", error));
			}
			const WindowsFileIdentity identity{
			        information.dwVolumeSerialNumber, information.nFileIndexHigh, information.nFileIndexLow};
			try
			{
				WindowsFileBackendDetail::AcquireRegistry(identity, access);
			}
			catch (...)
			{
				::CloseHandle(handle);
				throw;
			}

			OVERLAPPED overlapped{};
			DWORD flags = LOCKFILE_FAIL_IMMEDIATELY;
			if (access == FileAccess::ReadWrite)
			{
				flags |= LOCKFILE_EXCLUSIVE_LOCK;
			}
			if (!::LockFileEx(handle, flags, 0, MAXDWORD, MAXDWORD, &overlapped))
			{
				const DWORD error = ::GetLastError();
				WindowsFileBackendDetail::ReleaseRegistry(identity, access);
				::CloseHandle(handle);
				throw IOException("PersistentStore lock unavailable",
				                  WindowsFileBackendDetail::ErrorMessage("LockFileEx", error));
			}
			return WindowsFileBackend(handle, identity, access);
		}

		DWORD Transfer(const bool isWrite,
		               const UInt64 offset,
		               Byte* bytesPtr,
		               const DWORD byteCount) const
		{
			OVERLAPPED overlapped{};
			overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
			overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32U);
			DWORD transferred = 0;
			const BOOL hasStarted = isWrite ? ::WriteFile(m_Handle, bytesPtr, byteCount, &transferred, &overlapped)
			                                : ::ReadFile(m_Handle, bytesPtr, byteCount, &transferred, &overlapped);
			if (!hasStarted)
			{
				const DWORD error = ::GetLastError();
				if (error != ERROR_IO_PENDING
				    || !::GetOverlappedResult(m_Handle, &overlapped, &transferred, TRUE))
				{
					const DWORD finalError = error == ERROR_IO_PENDING ? ::GetLastError() : error;
					throw IOException(WindowsFileBackendDetail::ErrorMessage(
					        isWrite ? "WriteFile" : "ReadFile", finalError));
				}
			}
			return transferred;
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
			if (offset > static_cast<UInt64>(std::numeric_limits<LONGLONG>::max())
			    || byteCount > static_cast<UInt64>(std::numeric_limits<LONGLONG>::max()) - offset)
			{
				throw IOException("PersistentStore exact I/O exceeds the Windows offset range", operation);
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
			if (m_Handle != INVALID_HANDLE_VALUE)
			{
				::CloseHandle(m_Handle);
				m_Handle = INVALID_HANDLE_VALUE;
			}
			if (m_HasRegistryEntry)
			{
				WindowsFileBackendDetail::ReleaseRegistry(m_Identity, m_Access);
				m_HasRegistryEntry = false;
			}
		}

		HANDLE m_Handle = INVALID_HANDLE_VALUE;
		WindowsFileIdentity m_Identity;
		FileAccess m_Access = FileAccess::ReadOnly;
		bool m_HasRegistryEntry = false;
	};
}
