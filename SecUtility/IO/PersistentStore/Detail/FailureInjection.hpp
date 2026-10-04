// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Raw/Int.hpp>


namespace SecUtility::IO::PersistentStoreDetail
{
	enum class FailurePoint
	{
		RebuildAllocator,
		ReservePayload,
		GrowPayload,
		MapPayload,
		EncodePayload,
		CompletePayload,
		SerializeDirectory,
		ReserveDirectory,
		GrowDirectory,
		WriteDirectory,
		ReadBackDirectory,
		PrepareRuntimeState,
		BeginPublication,
		PublicationWriteProgress,
		ReadBackPublication,
		InstallRuntimeState,
		ExactReadProgress,
		ExactWriteProgress
	};

	using FailureHandler = void (*)(FailurePoint point, UInt64 detail, void* contextPtr);

	class ScopedFailureInjection final
	{
	public:
		ScopedFailureInjection(const FailureHandler handler, void* const contextPtr) noexcept
		    : m_PreviousHandler(Handler), m_PreviousContextPtr(ContextPtr)
		{
			Handler = handler;
			ContextPtr = contextPtr;
		}

		~ScopedFailureInjection() noexcept
		{
			Handler = m_PreviousHandler;
			ContextPtr = m_PreviousContextPtr;
		}

		ScopedFailureInjection(const ScopedFailureInjection&) = delete;
		ScopedFailureInjection& operator=(const ScopedFailureInjection&) = delete;

		static void Observe(const FailurePoint point, const UInt64 detail = 0)
		{
			if (Handler != nullptr) Handler(point, detail, ContextPtr);
		}

		static bool IsEnabled() noexcept { return Handler != nullptr; }

	private:
		inline static thread_local FailureHandler Handler = nullptr;
		inline static thread_local void* ContextPtr = nullptr;
		FailureHandler m_PreviousHandler;
		void* m_PreviousContextPtr;
	};
}
