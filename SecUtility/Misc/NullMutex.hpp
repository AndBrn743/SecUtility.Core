// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once


namespace SecUtility
{
	/// <summary>
	/// A zero-cost mutex that satisfies both <c>Lockable</c> and <c>SharedLockable</c>.
	/// </summary>
	/// <remarks>
	/// Intended as the <c>Mutex</c> template argument when synchronization is not required.
	/// See <c>BasicCachedFunction</c> for a usage example.
	/// With <c>[[no_unique_address]]</c> on the member, an empty <c>NullMutex</c> takes no storage.
	/// </remarks>
	class NullMutex
	{
	public:
		/// <summary>Acquires exclusive ownership of the mutex. No-op.</summary>
		constexpr void lock() noexcept
		{
			/* NO-OP */
		}

		/// <summary>Attempts to acquire exclusive ownership. Always succeeds.</summary>
		/// <returns>Always <c>true</c>.</returns>
		constexpr bool try_lock() noexcept
		{
			return true;
		}

		/// <summary>Releases exclusive ownership of the mutex. No-op.</summary>
		constexpr void unlock() noexcept
		{
			/* NO-OP */
		}

		/// <summary>Acquires shared ownership of the mutex. No-op.</summary>
		constexpr void lock_shared() noexcept
		{
			/* NO-OP */
		}

		/// <summary>Attempts to acquire shared ownership. Always succeeds.</summary>
		/// <returns>Always <c>true</c>.</returns>
		constexpr bool try_lock_shared() noexcept
		{
			return true;
		}

		/// <summary>Releases shared ownership of the mutex. No-op.</summary>
		constexpr void unlock_shared() noexcept
		{
			/* NO-OP */
		}
	};
}
