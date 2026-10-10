// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/IO/PersistentStore/Detail/FileBackend.hpp>
#include <SecUtility/IO/PersistentStore/Detail/Directory.hpp>

#include <memory>


namespace SecUtility::IO::PersistentStoreDetail
{
	class StoreState;

	class MappingLease final
	{
	public:
		MappingLease(std::shared_ptr<const StoreState> statePtr,
		             ReadOnlyMappedRegion region,
		             const ExtentIdentity identity) noexcept
		    : m_StatePtr(std::move(statePtr)), m_Region(std::move(region)), m_Identity(identity)
		{
			/* NO CODE */
		}
		~MappingLease() noexcept;

		const Byte* Data() const noexcept { return m_Region.Data(); }
		std::size_t Size() const noexcept { return m_Region.Size(); }

	private:
		std::shared_ptr<const StoreState> m_StatePtr;
		ReadOnlyMappedRegion m_Region;
		ExtentIdentity m_Identity;
	};
}
