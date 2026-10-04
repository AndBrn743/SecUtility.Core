// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/IO/PersistentStore/Detail/FileBackend.hpp>

#include <memory>


namespace SecUtility::IO::PersistentStoreDetail
{
#if defined(_WIN32)
	using ReadOnlyMappedRegion = WindowsReadOnlyMappedRegion;
	using ReadWriteMappedRegion = WindowsReadWriteMappedRegion;
#else
	using ReadOnlyMappedRegion = PosixReadOnlyMappedRegion;
	using ReadWriteMappedRegion = PosixReadWriteMappedRegion;
#endif

	class StoreState;

	class MappingLease final
	{
	public:
		MappingLease(std::shared_ptr<const StoreState> statePtr, ReadOnlyMappedRegion region) noexcept
		    : m_StatePtr(std::move(statePtr)), m_Region(std::move(region))
		{
			/* NO CODE */
		}

		const Byte* Data() const noexcept { return m_Region.Data(); }
		std::size_t Size() const noexcept { return m_Region.Size(); }

	private:
		std::shared_ptr<const StoreState> m_StatePtr;
		ReadOnlyMappedRegion m_Region;
	};
}
