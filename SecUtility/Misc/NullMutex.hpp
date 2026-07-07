// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once


namespace SecUtility
{
	class NullMutex
	{
	public:
		constexpr void lock() noexcept
		{
			/* NO CODE */
		}

		constexpr bool try_lock() noexcept
		{
			return true;
		}

		constexpr void unlock() noexcept
		{
			/* NO CODE */
		}

		constexpr void lock_shared() noexcept
		{
			/* NO CODE */
		}

		constexpr bool try_lock_shared() noexcept
		{
			return true;
		}

		constexpr void unlock_shared() noexcept
		{
			/* NO CODE */
		}
	};
}
