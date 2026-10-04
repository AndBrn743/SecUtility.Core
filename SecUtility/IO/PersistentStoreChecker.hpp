// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Raw/Int.hpp>

#include <filesystem>
#include <string>
#include <vector>


namespace SecUtility::IO
{
	struct PersistentStoreSlotReport
	{
		bool IsValid = false;
		UInt64 Generation = 0;
		std::string Reason;
	};

	struct PersistentStoreCheckReport
	{
		PersistentStoreSlotReport SlotA;
		PersistentStoreSlotReport SlotB;
		bool HasSelectedRoot = false;
		bool HasCrossRootConflict = false;
		UInt64 SelectedGeneration = 0;
		UInt64 LiveRecordCount = 0;
		UInt64 LivePayloadBytes = 0;
		UInt64 MetadataBytes = 0;
		UInt64 OtherRootOnlyBytes = 0;
		UInt64 ReclaimableBytes = 0;
		UInt64 PhysicalTailBytes = 0;
		UInt64 PhysicalFileBytes = 0;
		UInt64 CommittedFileBytes = 0;
		std::vector<std::string> Findings;
	};

	// Acquires the same nonblocking shared lock as OpenForReadOnly, validates both roots independently,
	// and reports space accounting without repairing or writing the file. A report with no selected
	// root still carries both rejection reasons; native open/read/lock failures throw IOException.
	PersistentStoreCheckReport CheckPersistentStore(const std::filesystem::path& path);
}

#include <SecUtility/IO/PersistentStore/Detail/Checker.hpp>


namespace SecUtility::IO
{
	inline PersistentStoreCheckReport CheckPersistentStore(const std::filesystem::path& path)
	{
		return PersistentStoreDetail::CheckPersistentStoreFile(path);
	}
}
