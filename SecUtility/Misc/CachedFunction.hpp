// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <mutex>
#include <shared_mutex>

#include <SecUtility/Meta/TypeTrait.hpp>
#include <SecUtility/Misc/NullMutex.hpp>


namespace SecUtility
{
	template <typename Mutex, typename... Args>
	class BasicCachedFunction;


	/// <summary>
	/// Generic memoizing wrapper that caches the results of a <see cref="Callable"/> keyed by
	/// argument tuple.
	/// </summary>
	/// <typeparam name="Mutex">Mutex policy. Must satisfy both <c>Lockable</c> and <c>SharedLockable</c>.
	/// Use <c>NullMutex</c> for unsynchronized access or <c>std::shared_mutex</c> for thread-safe access.</typeparam>
	/// <typeparam name="Callable">The callable type whose return values are cached. Reference and <c>void</c>
	/// return types are not supported.</typeparam>
	/// <remarks>
	/// <para>
	/// The cache lookup on a hit takes a <c>shared_lock</c>, so concurrent reads scale. A miss
	/// releases the shared lock, acquires an exclusive lock, re-checks (another thread may have
	/// inserted between the unlock and the lock), then calls the wrapped callable and stores the
	/// result. Concurrent misses on the same key therefore invoke the callable exactly once;
	/// concurrent misses on different keys serialize on the exclusive lock.
	/// </para>
	/// <para>
	/// Returned references are stable across subsequent inserts (<c>std::map</c> and
	/// <c>std::optional</c> do not invalidate references on insert). They are invalidated by
	/// <see cref="Clear"/>. Holding a reference while another thread calls <see cref="Clear"/> is undefined behavior.
	/// </para>
	/// <para>
	/// End users should prefer <see cref="CachedFunction"/> or <see cref="ConcurrentCachedFunction"/>
	/// over instantiating <c>BasicCachedFunction</c> directly.
	/// </para>
	/// </remarks>
	template <typename Mutex, typename Callable>
	class BasicCachedFunction<Mutex, Callable>
	{
		using Traits = FunctionTraits<Callable>;
		using Result = typename Traits::ReturnType;

		template <typename... Args>
		struct arg_storage;

		template <typename... Args>
		struct arg_storage<TypeTuple<Args...>>
		{
			static_assert((!std::is_bounded_array_v<std::remove_reference_t<Args>> && ...),
			              "Please use std::array<T, S> instead");
			static_assert((!std::is_unbounded_array_v<std::remove_reference_t<Args>> && ...),
			              "Unbounded arrays are not supported");
			static_assert((!std::is_pointer_v<std::remove_reference_t<Args>> && ...), "pointers are not supported");

			using type = std::tuple<std::decay_t<Args>...>;
		};

	public:
		static_assert(!std::is_reference_v<Result>, "CachedFunction does not support reference types");
		static_assert(!std::is_void_v<Result>, "CachedFunction does not support void return types");

		/// <summary>Constructs the cache, storing <paramref name="function"/> for later invocation.</summary>
		/// <param name="function">Callable whose return values will be memoized.</param>
		explicit BasicCachedFunction(Callable function) : m_Function(std::move(function))
		{
			/* NO CODE */
		}

		/// <summary>
		/// Returns the cached result for the given arguments, computing it via the wrapped
		/// callable on first access.
		/// </summary>
		/// <param name="args">Arguments to look up (or compute) in the cache.</param>
		/// <returns>Reference to the cached result. The reference is valid until another thread
		/// calls <see cref="Clear"/>.</returns>
		/// <remarks>
		/// Cache hit: takes a shared lock and returns the existing entry. Cache miss: releases the
		/// shared lock, acquires an exclusive lock, re-checks for a racing inserter, then invokes
		/// the wrapped callable and emplaces its result. The callable is invoked at most once per
		/// unique argument tuple.
		/// </remarks>
		template <typename... Args>
		const Result& operator()(Args&&... args) noexcept(
		        noexcept(std::declval<Callable>()(std::forward<Args>(args)...)))
		{
			static_assert(sizeof...(Args) == Traits::Arity);

			if constexpr (Traits::Arity == 0)
			{
				{
					std::shared_lock lock(m_Mutex);
					if (m_Cache.has_value())
					{
						return m_Cache.value();
					}
				}

				std::lock_guard lock(m_Mutex);
				if (!m_Cache.has_value())
				{
					m_Cache = m_Function();
				}

				return m_Cache.value();
			}
			else
			{
				auto argTuple = std::make_tuple(std::forward<Args>(args)...);

				{
					std::shared_lock lock(m_Mutex);
					const auto iterator = m_Cache.find(argTuple);
					if (iterator != m_Cache.end())
					{
						return iterator->second;
					}
				}

				std::lock_guard lock(m_Mutex);
				const auto iterator = m_Cache.find(argTuple);
				if (iterator != m_Cache.end())
				{
					return iterator->second;
				}

				return (m_Cache.emplace(std::move(argTuple), m_Function(std::forward<Args>(args)...))).first->second;
			}
		}

		/// <summary>
		/// Looks up the cached result for the given arguments without computing or inserting.
		/// </summary>
		/// <param name="args">Arguments to look up in the cache.</param>
		/// <returns>Reference to the cached result.</returns>
		/// <exception cref="std::out_of_range">Thrown when no entry exists for the given arguments.</exception>
		/// <remarks>Takes a shared lock; safe to call concurrently with other const calls.</remarks>
		template <typename... Args>
		const Result& operator()(Args&&... args) const
		{
			static_assert(sizeof...(Args) == Traits::Arity);

			std::shared_lock lock(m_Mutex);

			if constexpr (Traits::Arity == 0)
			{
				if (!m_Cache.has_value())
				{
					throw std::out_of_range("CachedFunction: Result not in cache. Call non-const operator() first.");
				}

				return m_Cache.value();
			}
			else
			{
				const auto iterator = m_Cache.find(std::make_tuple(std::forward<Args>(args)...));

				if (iterator == m_Cache.end())
				{
					throw std::out_of_range("CachedFunction: Result not in cache. Call non-const operator(...) first.");
				}

				return iterator->second;
			}
		}

		/// <summary>Removes all entries from the cache.</summary>
		/// <remarks>
		/// Acquires an exclusive lock; blocks concurrent readers and writers until complete.
		/// Invalidates all previously-returned references.
		/// </remarks>
		void Clear()
		{
			std::lock_guard lock(m_Mutex);
			if constexpr (Traits::Arity == 0)
			{
				m_Cache = std::nullopt;
			}
			else
			{
				m_Cache.clear();
			}
		}

		/// <summary>Returns the number of entries currently held in the cache.</summary>
		/// <returns><c>0</c> or <c>1</c> for nullary callables; the size of the underlying map otherwise.</returns>
		std::size_t Size() const
		{
			std::shared_lock lock(m_Mutex);
			if constexpr (Traits::Arity == 0)
			{
				return m_Cache.has_value() ? 1 : 0;
			}
			else
			{
				return m_Cache.size();
			}
		}

	private:
#if defined(_MSC_VER) && !defined(__clang__)
		[[msvc::no_unique_address]]
#else
		[[no_unique_address]]
#endif
		mutable Mutex m_Mutex;
		Callable m_Function;
		std::conditional_t<Traits::Arity == 0,
		                   std::optional<Result>,
		                   std::map<typename arg_storage<typename Traits::ArgTypeTuple>::type, Result>>
		        m_Cache;
	};


	/// <summary>
	/// Type-erased memoizing function. Single-threaded; not safe for concurrent access.
	/// </summary>
	/// <typeparam name="Result">Return type of the wrapped function.</typeparam>
	/// <typeparam name="...Args">Argument types of the wrapped function.</typeparam>
	/// <remarks>
	/// Convenience alias over <see cref="BasicCachedFunction&lt;Mutex, Callable&gt;"/> with
	/// <c>NullMutex</c>. Use <see cref="ConcurrentCachedFunction&lt;Result(Args...)&gt;"/> if multiple
	/// threads may invoke the cache concurrently.
	/// </remarks>
	template <typename... Args>
	class CachedFunction;

	template <typename Result, typename... Args>
	class CachedFunction<Result(Args...)> : public BasicCachedFunction<NullMutex, std::function<Result(Args...)>>
	{
		using Base = BasicCachedFunction<NullMutex, std::function<Result(Args...)>>;

	public:
		using Base::Base;
	};

	/// <summary>
	/// Memoizing wrapper preserving the wrapped callable's exact type. Single-threaded; not safe
	/// for concurrent access.
	/// </summary>
	/// <typeparam name="Callable">The callable type whose return values are cached. Reference and
	/// <c>void</c> return types are not supported.</typeparam>
	/// <remarks>
	/// Convenience alias over <see cref="BasicCachedFunction&lt;Mutex, Callable&gt;"/> with
	/// <c>NullMutex</c>. Use <see cref="ConcurrentCachedFunction&lt;Callable&gt;"/> if multiple threads
	/// may invoke the cache concurrently.
	/// </remarks>
	template <typename Callable>
	class CachedFunction<Callable> : public BasicCachedFunction<NullMutex, Callable>
	{
		using Base = BasicCachedFunction<NullMutex, Callable>;

	public:
		using Base::Base;
	};

	template <typename Callable>
	CachedFunction(Callable) -> CachedFunction<Callable>;

	/// <summary>
	/// Type-erased thread-safe memoizing function. Concurrent reads scale; concurrent writes serialize.
	/// </summary>
	/// <typeparam name="Result">Return type of the wrapped function.</typeparam>
	/// <typeparam name="...Args">Argument types of the wrapped function.</typeparam>
	/// <remarks>
	/// <para>
	/// Convenience alias over <see cref="BasicCachedFunction&lt;Mutex, Callable&gt;"/> with
	/// <c>std::shared_mutex</c>. Cache hits take a shared lock and may run concurrently across reader
	/// threads. Cache misses serialize on an exclusive lock; the wrapped function is invoked at most
	/// once per unique argument tuple.
	/// </para>
	/// <para>
	/// Returned references are stable across subsequent inserts but are invalidated by
	/// <see cref="BasicCachedFunction&lt;Mutex, Callable&gt;.Clear"/>. Holding a reference while another
	/// thread calls <c>Clear</c> is undefined behavior.
	/// </para>
	/// </remarks>
	template <typename... Args>
	class ConcurrentCachedFunction;

	template <typename Result, typename... Args>
	class ConcurrentCachedFunction<Result(Args...)> : public BasicCachedFunction<std::shared_mutex, std::function<Result(Args...)>>
	{
		using Base = BasicCachedFunction<std::shared_mutex, std::function<Result(Args...)>>;

	public:
		using Base::Base;
	};

	/// <summary>
	/// Thread-safe memoizing wrapper preserving the wrapped callable's exact type. Concurrent reads
	/// scale; concurrent writes serialize.
	/// </summary>
	/// <typeparam name="Callable">The callable type whose return values are cached. Reference and
	/// <c>void</c> return types are not supported.</typeparam>
	/// <remarks>
	/// <para>
	/// Convenience alias over <see cref="BasicCachedFunction&lt;Mutex, Callable&gt;"/> with
	/// <c>std::shared_mutex</c>. Cache hits take a shared lock and may run concurrently across reader
	/// threads. Cache misses serialize on an exclusive lock; the wrapped function is invoked at most
	/// once per unique argument tuple.
	/// </para>
	/// <para>
	/// Returned references are stable across subsequent inserts but are invalidated by
	/// <see cref="BasicCachedFunction&lt;Mutex, Callable&gt;.Clear"/>. Holding a reference while another
	/// thread calls <c>Clear</c> is undefined behavior.
	/// </para>
	/// </remarks>
	template <typename Callable>
	class ConcurrentCachedFunction<Callable> : public BasicCachedFunction<std::shared_mutex, Callable>
	{
		using Base = BasicCachedFunction<std::shared_mutex, Callable>;

	public:
		using Base::Base;
	};

	template <typename Callable>
	ConcurrentCachedFunction(Callable) -> ConcurrentCachedFunction<Callable>;
}
