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
#include <stdexcept>
#include <thread>
#include <vector>

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
		explicit ThreadPool(const std::size_t numThreads = std::thread::hardware_concurrency()) : m_Stop(false)
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
				if (m_Stop)
				{
					throw std::runtime_error("Cannot submit task to stopped ThreadPool");
				}
				m_Tasks.emplace([task = std::move(packagedTask)] { (*task)(); });
			}

			m_Condition.notify_one();
			return result;
		}

		~ThreadPool() noexcept
		{
			{
				std::lock_guard lock(m_Mutex);
				m_Stop = true;
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
					m_Condition.wait(lock, [this] { return m_Stop || !m_Tasks.empty(); });

					if (m_Stop && m_Tasks.empty())
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
		bool m_Stop;  // Protected by m_Mutex, no need for atomic
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
