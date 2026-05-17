// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

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
	class CachedFunction<Result(Args...)> : public CachedFunction<std::function<Result(Args...)>>
	{
		using Base = CachedFunction<std::function<Result(Args...)>>;

	public:
		using Base::Base;
	};


	template <typename Callable>
	class CachedFunction<Callable>
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

		explicit CachedFunction(Callable function) : m_Function(std::move(function))
		{
			/* NO CODE */
		}

		// the call operator is not thread-safe. user should guard the calls to it with synchronization primitives
		template <typename... Args>
		const Result& operator()(Args&&... args) noexcept(
		        noexcept(std::declval<Callable>()(std::forward<Args>(args)...)))
		{
			static_assert(sizeof...(Args) == Traits::Arity);

			if constexpr (Traits::Arity == 0)
			{
				if (!m_Cache.has_value())
				{
					m_Cache = m_Function();
				}

				return m_Cache.value();
			}
			else
			{
				auto argTuple = std::make_tuple(std::forward<Args>(args)...);
				const auto iterator = m_Cache.find(argTuple);

				if (iterator != m_Cache.end())
				{
					return iterator->second;
				}

				auto result = m_Function(std::forward<Args>(args)...);
				return (m_Cache.emplace(std::move(argTuple), result)).first->second;
			}
		}

		template <typename... Args>
		const Result& operator()(Args&&... args) const
		{
			static_assert(sizeof...(Args) == Traits::Arity);

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

		void Clear() noexcept
		{
			if constexpr (Traits::Arity == 0)
			{
				m_Cache = std::nullopt;
			}
			else
			{
				m_Cache.clear();
			}
		}

		std::size_t Size() const noexcept
		{
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
		Callable m_Function;
		std::conditional_t<Traits::Arity == 0,
		                   std::optional<Result>,
		                   std::map<typename arg_storage<typename Traits::ArgTypeTuple>::type, Result>>
		        m_Cache;
	};


	template <typename Callable>
	CachedFunction(Callable) -> CachedFunction<Callable>;
}
