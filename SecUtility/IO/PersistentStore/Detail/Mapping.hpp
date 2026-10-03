// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/IO/PersistentStore/Detail/FileBackend.hpp>


namespace SecUtility::IO::PersistentStoreDetail
{
#if defined(_WIN32)
	using ReadOnlyMappedRegion = WindowsReadOnlyMappedRegion;
	using WritableMappedRegion = WindowsWritableMappedRegion;
#else
	using ReadOnlyMappedRegion = PosixReadOnlyMappedRegion;
	using WritableMappedRegion = PosixWritableMappedRegion;
#endif
}
