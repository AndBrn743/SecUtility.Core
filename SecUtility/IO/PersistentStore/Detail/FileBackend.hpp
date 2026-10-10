// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FailureInjection.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Format.hpp>
#include <SecUtility/IO/RandomAccessFile.hpp>
#include <SecUtility/Raw/Int.hpp>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <utility>


namespace SecUtility::IO::PersistentStoreDetail
{
	enum class FileAccess
	{
		ReadOnly,
		ReadWrite
	};

	class FileBackend final
	{
	public:
		static FileBackend Open(const std::filesystem::path& path, const FileAccess access)
		{
			RandomAccessFile file = access == FileAccess::ReadOnly
			                                ? RandomAccessFile::OpenForReadOnly(path)
			                                : RandomAccessFile::OpenForReadWrite(path);
			return AdoptAndLock(std::move(file), access, path);
		}

		static std::optional<FileBackend> TryCreateExclusive(const std::filesystem::path& path)
		{
			auto file = RandomAccessFile::TryCreateExclusive(path);
			if (!file.has_value()) return std::nullopt;
			return AdoptAndLock(std::move(*file), FileAccess::ReadWrite, path);
		}

		FileBackend(FileBackend&&) noexcept = default;
		FileBackend& operator=(FileBackend&&) noexcept = default;
		FileBackend(const FileBackend&) = delete;
		FileBackend& operator=(const FileBackend&) = delete;
		~FileBackend() noexcept = default;

		UInt64 GetPhysicalFileBytes() const
		{
			try { return m_File.Size(); }
			catch (const Exception& exception) { throw IOException("PersistentStore file-size query failed", exception.what()); }
		}

		void SetPhysicalFileBytes(const UInt64 byteCount)
		{
			try { m_File.Resize(byteCount); }
			catch (const InvalidOperationException&) { throw; }
			catch (const Exception& exception) { throw IOException("PersistentStore file resize failed", exception.what()); }
		}

		void ReadExact(const UInt64 offset, Byte* out_bytesPtr, const std::size_t byteCount) const
		{
			ScopedFailureInjection::Observe(FailurePoint::ExactReadProgress, 0);
			try { m_File.ReadExactAt(offset, out_bytesPtr, byteCount); }
			catch (const Exception& exception) { throw IOException("PersistentStore exact read failed", exception.what()); }
			ScopedFailureInjection::Observe(FailurePoint::ExactReadProgress, byteCount);
		}

		void WriteExact(const UInt64 offset, const Byte* bytesPtr, const std::size_t byteCount)
		{
			ScopedFailureInjection::Observe(FailurePoint::ExactWriteProgress, 0);
			try { m_File.WriteExactAt(offset, bytesPtr, byteCount); }
			catch (const InvalidOperationException&) { throw; }
			catch (const Exception& exception) { throw IOException("PersistentStore exact write failed", exception.what()); }
			ScopedFailureInjection::Observe(FailurePoint::ExactWriteProgress, byteCount);
		}

		FileAccess GetAccess() const noexcept { return m_Access; }

		bool DoesPathIdentifySameFile(const std::filesystem::path& path) const noexcept
		{
			return m_File.IdentifiesSameFile(path);
		}

		ReadOnlyMappedRegion MapReadOnly(const UInt64 offset,
		                                   const std::size_t byteCount,
		                                   const UInt64 alignment) const
		{
			ValidateAlignment(offset, alignment);
			try
			{
				auto region = m_File.MapReadOnly(offset, byteCount);
				ValidateMappedAddress(region.Data(), alignment);
				return region;
			}
			catch (const IOException&) { throw; }
			catch (const Exception& exception) { throw IOException("PersistentStore read-only mapping failed", exception.what()); }
		}

		ReadWriteMappedRegion MapReadWrite(const UInt64 offset,
		                                     const std::size_t byteCount,
		                                     const UInt64 alignment)
		{
			ValidateAlignment(offset, alignment);
			try
			{
				auto region = m_File.MapReadWrite(offset, byteCount);
				ValidateMappedAddress(region.Data(), alignment);
				return region;
			}
			catch (const InvalidOperationException&) { throw; }
			catch (const IOException&) { throw; }
			catch (const Exception& exception) { throw IOException("PersistentStore read/write mapping failed", exception.what()); }
		}

	private:
		FileBackend(RandomAccessFile file, RandomAccessFile::Lock lock, const FileAccess access) noexcept
		    : m_File(std::move(file)), m_Lock(std::move(lock)), m_Access(access)
		{
			/* NO CODE */
		}

		static FileBackend AdoptAndLock(RandomAccessFile file,
		                                const FileAccess access,
		                                const std::filesystem::path& path)
		{
			const auto mode = access == FileAccess::ReadOnly ? RandomAccessFile::LockMode::Shared
			                                                     : RandomAccessFile::LockMode::Exclusive;
			auto lock = file.TryAcquireLock(mode);
			if (!lock.has_value()) throw IOException("PersistentStore lock unavailable", path.string());
			return FileBackend(std::move(file), std::move(*lock), access);
		}

		static void ValidateAlignment(const UInt64 offset, const UInt64 alignment)
		{
			if (alignment == 0 || alignment > MaximumPayloadAlignment || (alignment & (alignment - 1)) != 0)
				throw IOException("PersistentStore mapping alignment must be a power of two from 1 through 4096");
			if (offset % alignment != 0)
				throw IOException("PersistentStore mapping offset does not satisfy the requested alignment");
		}

		template <typename ByteType>
		static void ValidateMappedAddress(ByteType* bytesPtr, const UInt64 alignment)
		{
			if (bytesPtr != nullptr && reinterpret_cast<std::uintptr_t>(bytesPtr) % alignment != 0)
				throw IOException("PersistentStore mapped address does not satisfy the requested alignment");
		}

		RandomAccessFile m_File;
		RandomAccessFile::Lock m_Lock;
		FileAccess m_Access;
	};
}
