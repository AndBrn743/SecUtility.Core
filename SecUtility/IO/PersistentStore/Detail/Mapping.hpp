// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/IO/PersistentStore/Detail/FileBackend.hpp>


namespace SecUtility::IO::PersistentStoreDetail
{
#if defined(_WIN32)
	using ReadOnlyMappedRegion = WindowsReadOnlyMappedRegion;
	using ReadWriteMappedRegion = WindowsReadWriteMappedRegion;
#else
	using ReadOnlyMappedRegion = PosixReadOnlyMappedRegion;
	using ReadWriteMappedRegion = PosixReadWriteMappedRegion;
#endif
}
