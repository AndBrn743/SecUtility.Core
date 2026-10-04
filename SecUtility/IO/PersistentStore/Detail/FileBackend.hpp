// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/IO/PersistentStore/Detail/FailureInjection.hpp>
#include <SecUtility/Raw/Int.hpp>

#include <cstddef>
#include <utility>


namespace SecUtility::IO::PersistentStoreDetail
{
	enum class FileAccess
	{
		ReadOnly,
		ReadWrite
	};

	struct ExactTransferResult
	{
		std::size_t Bytes = 0;
		bool ShouldRetry = false;
	};

	template <typename TByte, typename Operation>
	void CompleteExactTransfer(const UInt64 offset,
	                           TByte* bytesPtr,
	                           const std::size_t byteCount,
	                           const bool isRead,
	                           Operation&& operation)
	{
		std::size_t completed = 0;
		while (completed < byteCount)
		{
			ScopedFailureInjection::Observe(
			        isRead ? FailurePoint::ExactReadProgress : FailurePoint::ExactWriteProgress, completed);
			const ExactTransferResult result = operation(
			        offset + completed, bytesPtr + completed, byteCount - completed);
			if (result.ShouldRetry)
			{
				continue;
			}
			if (result.Bytes == 0)
			{
				throw IOException(isRead ? "unexpected end of PersistentStore file during exact read"
				                         : "zero-length result during exact PersistentStore write");
			}
			if (result.Bytes > byteCount - completed)
			{
				throw IOException("PersistentStore backend reported more transferred bytes than requested");
			}
			completed += result.Bytes;
			ScopedFailureInjection::Observe(
			        isRead ? FailurePoint::ExactReadProgress : FailurePoint::ExactWriteProgress, completed);
		}
	}
}

#if defined(_WIN32)
#include <SecUtility/IO/PersistentStore/Detail/WindowsFileBackend.hpp>
namespace SecUtility::IO::PersistentStoreDetail
{
	using FileBackend = WindowsFileBackend;
}
#else
#include <SecUtility/IO/PersistentStore/Detail/PosixFileBackend.hpp>
namespace SecUtility::IO::PersistentStoreDetail
{
	using FileBackend = PosixFileBackend;
}
#endif
