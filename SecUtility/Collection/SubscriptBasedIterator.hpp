// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <cassert>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>


namespace SecUtility
{
	template <typename Iterable, typename Subscript = std::size_t, typename SubscriptOffset = std::ptrdiff_t>
	class SubscriptBasedIterator
	{
		Iterable* m_CollectionPtr{};
		Subscript m_Index{};


	public:
		using iterator_category = std::random_access_iterator_tag;
		using difference_type = SubscriptOffset;

		using reference = decltype(std::declval<Iterable&>()[std::declval<Subscript>()]);
		using value_type = std::remove_cv_t<std::remove_reference_t<reference>>;
		using pointer = std::conditional_t<std::is_lvalue_reference_v<reference>,
		                                   std::add_pointer_t<std::remove_reference_t<reference>>,
		                                   void>;

		constexpr SubscriptBasedIterator() noexcept = default;


		constexpr SubscriptBasedIterator(Iterable& collection, const Subscript index) noexcept
		    : m_CollectionPtr(std::addressof(collection)), m_Index(index)
		{
			/* NO CODE */
		}

		constexpr decltype(auto) operator*() const noexcept(noexcept((*m_CollectionPtr)[m_Index]))
		{
			return (*m_CollectionPtr)[m_Index];
		}

		template <typename TRef = reference>
		constexpr std::enable_if_t<std::is_lvalue_reference_v<TRef>, std::add_pointer_t<std::remove_reference_t<TRef>>>
		operator->() const noexcept(noexcept(std::addressof(operator*())))
		{
			return std::addressof(operator*());
		}

		constexpr SubscriptBasedIterator& operator+=(const SubscriptOffset offset) noexcept
		{
			m_Index += offset;
			return *this;
		}

		constexpr SubscriptBasedIterator& operator-=(const SubscriptOffset offset) noexcept
		{
			m_Index -= offset;
			return *this;
		}

		friend constexpr SubscriptBasedIterator operator+(SubscriptBasedIterator iterator,
		                                                  const SubscriptOffset offset) noexcept
		{
			iterator += offset;
			return iterator;
		}

		friend constexpr SubscriptBasedIterator operator+(SubscriptOffset offset,
		                                                  SubscriptBasedIterator iterator) noexcept
		{
			iterator += offset;
			return iterator;
		}

		friend constexpr SubscriptBasedIterator operator-(SubscriptBasedIterator iterator,
		                                                  const SubscriptOffset offset) noexcept
		{
			iterator -= offset;
			return iterator;
		}

		constexpr SubscriptBasedIterator& operator++() noexcept
		{
			++m_Index;
			return *this;
		}

		constexpr SubscriptBasedIterator& operator--() noexcept
		{
			--m_Index;
			return *this;
		}

		constexpr SubscriptBasedIterator operator++(int) noexcept
		{
			auto old = *this;
			m_Index++;
			return old;
		}

		constexpr SubscriptBasedIterator operator--(int) noexcept
		{
			auto old = *this;
			m_Index--;
			return old;
		}

		constexpr decltype(auto) operator[](const SubscriptOffset offset) const
		        noexcept(noexcept((*m_CollectionPtr)[m_Index + offset]))
		{
			return (*m_CollectionPtr)[m_Index + offset];
		}

		friend constexpr bool operator==(const SubscriptBasedIterator& lhs, const SubscriptBasedIterator& rhs) noexcept
		{
			return lhs.m_CollectionPtr == rhs.m_CollectionPtr && lhs.m_Index == rhs.m_Index;
		}

		friend constexpr bool operator!=(const SubscriptBasedIterator& lhs, const SubscriptBasedIterator& rhs) noexcept
		{
			return !(lhs == rhs);
		}

#define SEC_DEFINE_COMPARISON_OPERATOR(OP)                                                                             \
	friend constexpr bool operator OP(const SubscriptBasedIterator& lhs, const SubscriptBasedIterator& rhs) noexcept   \
	{                                                                                                                  \
		assert(lhs.m_CollectionPtr == rhs.m_CollectionPtr);                                                            \
		return lhs.m_Index OP rhs.m_Index;                                                                             \
	}

		SEC_DEFINE_COMPARISON_OPERATOR(>)
		SEC_DEFINE_COMPARISON_OPERATOR(<)
		SEC_DEFINE_COMPARISON_OPERATOR(>=)
		SEC_DEFINE_COMPARISON_OPERATOR(<=)

#undef SEC_DEFINE_COMPARISON_OPERATOR

		friend constexpr SubscriptOffset operator-(const SubscriptBasedIterator& lhs,
		                                           const SubscriptBasedIterator& rhs) noexcept
		{
			assert(lhs.m_CollectionPtr == rhs.m_CollectionPtr);
			return static_cast<SubscriptOffset>(lhs.m_Index) - static_cast<SubscriptOffset>(rhs.m_Index);
		}
	};
}
