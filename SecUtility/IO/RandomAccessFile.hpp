// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Misc/Export.hpp>
#include <SecUtility/Raw/Int.hpp>

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <utility>


namespace SecUtility::IO
{
	namespace RandomAccessFileDetail
	{
		struct FileState;
		struct ReadOnlyMappingState;
		struct ReadWriteMappingState;
	}

	class SECUTILITY_CORE_API FileIdentity final
	{
	public:
		friend bool operator==(const FileIdentity& left, const FileIdentity& right) noexcept
		{
			return left.m_Components == right.m_Components;
		}

		friend bool operator!=(const FileIdentity& left, const FileIdentity& right) noexcept
		{
			return !(left == right);
		}

	private:
		friend class RandomAccessFile;
		explicit FileIdentity(std::array<UInt64, 3> components) noexcept : m_Components(components) {}
		std::array<UInt64, 3> m_Components{};
	};

	class SECUTILITY_CORE_API ReadOnlyMappedRegion final
	{
	public:
		ReadOnlyMappedRegion(ReadOnlyMappedRegion&&) noexcept;
		ReadOnlyMappedRegion& operator=(ReadOnlyMappedRegion&&) noexcept;
		ReadOnlyMappedRegion(const ReadOnlyMappedRegion&) = delete;
		ReadOnlyMappedRegion& operator=(const ReadOnlyMappedRegion&) = delete;
		~ReadOnlyMappedRegion() noexcept;

		const Byte* Data() const noexcept;
		std::size_t Size() const noexcept;

	private:
		friend class RandomAccessFile;
		explicit ReadOnlyMappedRegion(std::unique_ptr<RandomAccessFileDetail::ReadOnlyMappingState> statePtr) noexcept;
		std::unique_ptr<RandomAccessFileDetail::ReadOnlyMappingState> m_StatePtr;
	};

	class SECUTILITY_CORE_API ReadWriteMappedRegion final
	{
	public:
		ReadWriteMappedRegion(ReadWriteMappedRegion&&) noexcept;
		ReadWriteMappedRegion& operator=(ReadWriteMappedRegion&&) noexcept;
		ReadWriteMappedRegion(const ReadWriteMappedRegion&) = delete;
		ReadWriteMappedRegion& operator=(const ReadWriteMappedRegion&) = delete;
		~ReadWriteMappedRegion() noexcept;

		Byte* Data() const noexcept;
		std::size_t Size() const noexcept;
		void Complete();

	private:
		friend class RandomAccessFile;
		explicit ReadWriteMappedRegion(std::unique_ptr<RandomAccessFileDetail::ReadWriteMappingState> statePtr) noexcept;
		std::unique_ptr<RandomAccessFileDetail::ReadWriteMappingState> m_StatePtr;
	};

	class SECUTILITY_CORE_API RandomAccessFile final
	{
	public:
		enum class LockMode
		{
			Shared,
			Exclusive
		};

		class SECUTILITY_CORE_API Lock final
		{
		public:
			Lock(Lock&&) noexcept;
			Lock& operator=(Lock&&) noexcept;
			Lock(const Lock&) = delete;
			Lock& operator=(const Lock&) = delete;
			~Lock() noexcept;

		private:
			friend class RandomAccessFile;
			Lock(std::shared_ptr<RandomAccessFileDetail::FileState> statePtr, LockMode mode) noexcept;
			void Reset() noexcept;

			std::shared_ptr<RandomAccessFileDetail::FileState> m_StatePtr;
			LockMode m_Mode = LockMode::Shared;
		};

		static RandomAccessFile OpenForReadOnly(const std::filesystem::path& path);
		static RandomAccessFile OpenForReadWrite(const std::filesystem::path& path);
		static std::optional<RandomAccessFile> TryCreateExclusive(const std::filesystem::path& path);

		RandomAccessFile(RandomAccessFile&&) noexcept = default;
		RandomAccessFile& operator=(RandomAccessFile&&) noexcept = default;
		RandomAccessFile(const RandomAccessFile&) = delete;
		RandomAccessFile& operator=(const RandomAccessFile&) = delete;
		~RandomAccessFile() noexcept = default;

		UInt64 Size() const;
		void Resize(UInt64 bytes);
		void ReadExactAt(UInt64 offset, Byte* out_bytesPtr, std::size_t bytes) const;
		void WriteExactAt(UInt64 offset, const Byte* bytesPtr, std::size_t bytes);
		ReadOnlyMappedRegion MapReadOnly(UInt64 offset, std::size_t bytes) const;
		ReadWriteMappedRegion MapReadWrite(UInt64 offset, std::size_t bytes);
		FileIdentity GetIdentity() const;
		bool IdentifiesSameFile(const std::filesystem::path& path) const noexcept;
		std::optional<Lock> TryAcquireLock(LockMode mode) const;

	private:
		explicit RandomAccessFile(std::shared_ptr<RandomAccessFileDetail::FileState> statePtr) noexcept
		    : m_StatePtr(std::move(statePtr))
		{
			/* NO CODE */
		}

		std::shared_ptr<RandomAccessFileDetail::FileState> m_StatePtr;
	};
}

#if !defined(SECUTILITY_CORE_COMPILED)
#include <SecUtility/IO/Detail/RandomAccessFileImpl.hpp>
#endif
