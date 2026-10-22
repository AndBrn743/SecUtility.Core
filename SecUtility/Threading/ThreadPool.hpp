// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Andy Brown

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <SecUtility/Diagnostic/Exception.hpp>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#include <sanitizer/lsan_interface.h>
#define LSAN_IGNORE(ptr) __lsan_ignore_object(ptr)
#endif
#endif

#ifndef LSAN_IGNORE
#define LSAN_IGNORE(ptr) ((void)0)
#endif


namespace SecUtility::Threading
{
	class ThreadPool
	{
	public:
		explicit ThreadPool(const std::size_t numThreads = std::thread::hardware_concurrency()) : m_HasStopped(false)
		{
			const auto threadCount = std::max(numThreads, std::size_t{1});
			m_Workers.reserve(threadCount);

			for (std::size_t i = 0; i < threadCount; ++i)
			{
				m_Workers.emplace_back([this] { WorkerThread(); });
			}
		}

		ThreadPool(const ThreadPool&) = delete;
		ThreadPool(ThreadPool&&) = delete;
		ThreadPool& operator=(const ThreadPool&) = delete;
		ThreadPool& operator=(ThreadPool&&) = delete;

		template <typename Func, typename... Args>
		auto Submit(Func&& func, Args&&... args)
		{
			using ReturnType = std::invoke_result_t<std::decay_t<Func>, std::decay_t<Args>...>;

			auto packagedTask = std::make_shared<std::packaged_task<ReturnType()>>(
#if defined(__cpp_init_captures) && __cpp_init_captures >= 201803L
			        [func = std::forward<Func>(func), ... args = std::forward<Args>(args)]() mutable -> ReturnType
			        { return std::invoke(std::move(func), std::move(args)...); }
#else
			        [func = std::forward<Func>(func),
			         tup = std::make_tuple(std::forward<Args>(args)...)]() mutable -> ReturnType
			        { return std::apply(func, std::move(tup)); }
#endif
			);

			std::future<ReturnType> result = packagedTask->get_future();

			{
				std::lock_guard lock(m_Mutex);
				if (m_HasStopped)
				{
					throw InvalidOperationException("Cannot submit task to stopped ThreadPool");
				}
				m_Tasks.emplace([task = std::move(packagedTask)] { (*task)(); });
			}

			m_Condition.notify_one();
			return result;
		}

		/**
		 * Submits work whose result and exceptions will not be observed through a future.
		 * An exception escaping the submitted callable terminates the process.
		 */
		template <typename Func, typename... Args>
		void SubmitDetached(Func&& func, Args&&... args)
		{
#if defined(__cpp_init_captures) && __cpp_init_captures >= 201803L
			auto callable = [func = std::forward<Func>(func), ... args = std::forward<Args>(args)]() mutable
			{ std::invoke(std::move(func), std::move(args)...); };
#else
			auto callable =
			        [func = std::forward<Func>(func), tup = std::make_tuple(std::forward<Args>(args)...)]() mutable
			{ std::apply(std::move(func), std::move(tup)); };
#endif

			{
				auto task = std::make_shared<decltype(callable)>(std::move(callable));
				std::lock_guard lock(m_Mutex);
				if (m_HasStopped)
				{
					throw InvalidOperationException("Cannot submit task to stopped ThreadPool");
				}
				m_Tasks.emplace([task = std::move(task)] { (*task)(); });
			}

			m_Condition.notify_one();
		}

		~ThreadPool() noexcept
		{
			{
				std::lock_guard lock(m_Mutex);
				m_HasStopped = true;
			}
			m_Condition.notify_all();

			for (auto& worker : m_Workers)
			{
				if (worker.joinable())
				{
					worker.join();
				}
			}
		}

		std::size_t ThreadCount() const noexcept
		{
			return m_Workers.size();
		}

	private:
		void WorkerThread()
		{
			while (true)
			{
				std::function<void()> task;

				{
					std::unique_lock lock(m_Mutex);
					m_Condition.wait(lock, [this] { return m_HasStopped || !m_Tasks.empty(); });

					if (m_HasStopped && m_Tasks.empty())
					{
						return;
					}

					task = std::move(m_Tasks.front());
					m_Tasks.pop();
				}

				// Execute task outside lock
				if (static_cast<bool>(task))
				{
					task();
				}
			}
		}

		std::vector<std::thread> m_Workers;
		std::queue<std::function<void()>> m_Tasks;
		mutable std::mutex m_Mutex;  // mutable for const methods
		std::condition_variable m_Condition;
		bool m_HasStopped;  // Protected by m_Mutex, no need for atomic
	};


	/// <summary>
	/// Returns the application-wide master thread pool.
	/// </summary>
	/// <remarks>
	/// This thread pool is never destroyed to avoid static destruction order issues. Threads are terminated by the OS
	/// at process exit. The pool is created on first use with hardware_concurrency threads.
	/// </remarks>
	inline ThreadPool& MasterThreadPool()
	{
		// Intentional leak to avoid static destruction order issues
		// NOLINTNEXTLINE(cppcoreguidelines-owning-memory, cppcoreguidelines-avoid-non-const-global-variables)
		static ThreadPool* instance = []
		{
			auto ptr = std::make_unique<ThreadPool>(std::thread::hardware_concurrency());
			auto* raw = ptr.release();
			LSAN_IGNORE(raw);
			return raw;  // Explicit transfer of ownership to "immortal" scope
		}();
		return *instance;
	}
}  // namespace SecUtility::Threading

#if defined(LSAN_IGNORE)
#undef LSAN_IGNORE
#endif
