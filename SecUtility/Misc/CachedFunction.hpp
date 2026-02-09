//
// Created by Andy on 5/17/2023.
//

#pragma once

#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <SecUtility/Meta/TypeTrait.hpp>


namespace SecUtility
{
	template <typename... Args>
	class CachedFunction;


	template <typename Result, typename... Args>
	class CachedFunction<Result(Args...)>
	{
	public:
		static_assert(sizeof...(Args) != 0);
		static_assert(!std::is_reference_v<Result>, "CachedFunction does not support reference types");
		static_assert(!std::is_void_v<Result>, "CachedFunction does not support void return types");
		static_assert((!std::is_bounded_array_v<std::remove_reference_t<Args>> && ...),
		              "Please use std::array<T, S> instead");
		static_assert((!std::is_unbounded_array_v<std::remove_reference_t<Args>> && ...),
		              "Unbounded arrays are not supported");

		using Function = std::function<Result(Args...)>;

		explicit CachedFunction(Function function) : m_Function(std::move(function))
		{
			/* NO CODE */
		}

		// the call operator is not thread-safe. user should guard the calls to it with synchronization primitives
		template <typename... ArgsToo>
		const Result& operator()(ArgsToo&&... args)
		{
			auto argTuple = std::make_tuple(std::forward<ArgsToo>(args)...);
			const auto iterator = m_Cache.find(argTuple);

			if (iterator != m_Cache.end())
			{
				return iterator->second;
			}

			auto result = m_Function(std::forward<ArgsToo>(args)...);
			return (m_Cache.emplace(std::move(argTuple), result)).first->second;
		}

		template <typename... ArgsToo>
		const Result& operator()(ArgsToo&&... args) const
		{
			const auto iterator = m_Cache.find(std::make_tuple(std::forward<ArgsToo>(args)...));

			if (iterator == m_Cache.end())
			{
				throw std::out_of_range("CachedFunction: Result not in cache. Call non-const operator(...) first.");
			}

			return iterator->second;
		}

		void Clear() noexcept
		{
			m_Cache.clear();
		}

		size_t Size() const noexcept
		{
			return m_Cache.size();
		}

	private:
		Function m_Function;
		std::map<std::tuple<std::decay_t<Args>...>, Result> m_Cache;
	};


	template <typename TResult>
	class CachedFunction<TResult()>
	{
	public:
		static_assert(!std::is_reference_v<TResult>, "CachedFunction does not support reference types");
		static_assert(!std::is_void_v<TResult>, "CachedFunction does not support void return types");

		using Function = std::function<TResult()>;

		explicit CachedFunction(Function function) : m_Function(std::move(function))
		{
			/* NO CODE */
		}

		// the call operator is not thread-safe. user should guard the calls to it with synchronization primitives
		const TResult& operator()()
		{
			if (!m_Cache.has_value())
			{
				m_Cache = m_Function();
			}

			return m_Cache.value();
		}

		const TResult& operator()() const
		{
			if (!m_Cache.has_value())
			{
				throw std::out_of_range("CachedFunction: Result not in cache. Call non-const operator() first.");
			}

			return m_Cache.value();
		}

		void Clear() noexcept
		{
			m_Cache = std::nullopt;
		}

		size_t Size() const noexcept
		{
			return m_Cache.has_value() ? 1 : 0;
		}

	private:
		Function m_Function;
		std::optional<TResult> m_Cache;
	};
}  // namespace SecUtility
