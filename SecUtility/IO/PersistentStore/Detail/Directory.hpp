// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Raw/Int.hpp>

#include <string>
#include <vector>


namespace SecUtility::IO::PersistentStoreDetail
{
	struct Extent
	{
		UInt64 Offset = 0;
		UInt64 Capacity = 0;
	};

	struct Allocation
	{
		Extent AllocatedExtent;
		UInt64 PayloadOffset = 0;
	};

	struct DirectoryEntry
	{
		std::string Key;
		Extent AllocatedExtent;
		UInt64 PayloadOffset = 0;
		UInt64 PayloadBytes = 0;
	};

	struct Directory
	{
		UInt64 Generation = 0;
		std::vector<DirectoryEntry> Entries;
	};
}
