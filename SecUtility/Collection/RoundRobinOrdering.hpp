// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Andy Brown

#pragma once

#include <SecUtility/Collection/SubscriptBasedIterator.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>
#include <SecUtility/Raw/Int.hpp>
#include <utility>
#include <vector>


namespace SecUtility
{
	template <typename Index>
	class RoundRobinOrderingGenerator
	{
	public:
		class RoundRobinOrderingPairs : public std::vector<std::pair<Index, Index>>
		{
			using Base = std::vector<std::pair<Index, Index>>;

		public:
			using Base::Base;
			using Base::operator=;

			friend std::ostream& operator<<(std::ostream& os, const RoundRobinOrderingPairs& pairs)
			{
				os << '{';

				if (!pairs.empty())
				{
					os << '(' << pairs[0].first << ", " << pairs[0].second << ')';

					for (auto iterator = pairs.begin() + 1; iterator != pairs.end(); iterator++)
					{
						os << ", (" << iterator->first << ", " << iterator->second << ')';
					}
				}

				return os << '}';
			}
		};

		static RoundRobinOrderingPairs GenerateInitialPairs(const Index size,
		                                                    Index initialValue = 0,
		                                                    const Index invalidValue = static_cast<Index>(-1))
		{
			const auto n = (size + 1) / 2;
			RoundRobinOrderingPairs result(n);

			for (auto& pair : result)
			{
				pair = {initialValue, initialValue + n};
				initialValue += 1;
			}

			if (size % 2 != 0)
			{
				result.back().second = invalidValue;
			}

			return result;
		}

		static RoundRobinOrderingPairs GenerateNextPairsFrom(const RoundRobinOrderingPairs& pairs)
		{
			const auto n = pairs.size();
			RoundRobinOrderingPairs result(n);

#if defined(_OPENMP)
#pragma omp parallel for default(none) shared(n, pairs, result) schedule(static)
#endif
			for (std::size_t k = 0; k < n; k++)
			{
				if (k == 0)
				{
					result[k].first = pairs[k].first;
				}
				else if (k == 1)
				{
					result[k].first = pairs[k - 1].second;
				}
				else
				{
					result[k].first = pairs[k - 1].first;
				}

				if (k == n - 1)
				{
					result[k].second = pairs[k].first;
				}
				else
				{
					result[k].second = pairs[k + 1].second;
				}
			}

			return result;
		}

		static void UpdatePairs(RoundRobinOrderingPairs& ref_pairs)
		{
			ref_pairs = GenerateNextPairsFrom(ref_pairs);
		}
	};


	template <typename Derived>
	class AbstractRoundRobinOrdering
	{
		friend Derived;

	public:
		/* CRTP VIRTUAL */ void NextCycle() noexcept
		{
			AsDerived().NextCycle();
		}

		/* CRTP VIRTUAL */ auto PairsPerCycle() const noexcept
		{
			return AsDerived().PairsPerCycle();
		}

		/* CRTP VIRTUAL */ auto Period() const noexcept
		{
			return AsDerived().Period();
		}

		decltype(auto) operator[](const Int64 index) const noexcept(noexcept(std::declval<Derived>()[index]))
		{
			return AsDerived()[index];
		}

		using Iterator = SubscriptBasedIterator<Derived, Int64>;

		Iterator begin()
		{
			return {AsDerived(), 0};
		}

		Iterator end()
		{
			return {AsDerived(), PairsPerCycle()};
		}


	private:
		constexpr AbstractRoundRobinOrdering() = default;
		constexpr AbstractRoundRobinOrdering(const AbstractRoundRobinOrdering&) noexcept = default;
		constexpr AbstractRoundRobinOrdering(AbstractRoundRobinOrdering&&) noexcept = default;
		constexpr AbstractRoundRobinOrdering& operator=(const AbstractRoundRobinOrdering&) noexcept = default;
		constexpr AbstractRoundRobinOrdering& operator=(AbstractRoundRobinOrdering&&) noexcept = default;
		~AbstractRoundRobinOrdering() noexcept = default;

		Derived& AsDerived() noexcept
		{
			return *static_cast<Derived*>(this);
		}

		const Derived& AsDerived() const noexcept
		{
			return *static_cast<const Derived*>(this);
		}
	};


	template <typename Range>
	class RoundRobinOrdering : public AbstractRoundRobinOrdering<RoundRobinOrdering<Range>>
	{
	private:
		using Base = AbstractRoundRobinOrdering<RoundRobinOrdering<Range>>;
		using Item = std::decay_t<decltype(*std::declval<Range>().begin())>;
		Range m_Range;
		Item m_InvalidItemMarker;
		Int64 m_RangeSize;
		Int64 m_EffectiveAndActiveRangeSize;
		Int64 m_PairsPerCycle;
		Int64 m_Period;
		Int64 m_RobinIndex;

	public:
		RoundRobinOrdering(Range range, const Item invalidItemMarker)
		    : m_Range(std::move(range)), m_InvalidItemMarker(invalidItemMarker),
		      m_RangeSize(std::distance(std::begin(m_Range), std::end(m_Range))),
		      m_EffectiveAndActiveRangeSize(m_RangeSize % 2 == 0 ? m_RangeSize - 1 : m_RangeSize),
		      m_PairsPerCycle(m_RangeSize % 2 == 0 ? m_RangeSize / 2 : m_RangeSize / 2 + 1),
		      m_Period(m_RangeSize * (m_RangeSize - 1) / 2 / (m_RangeSize / 2)), m_RobinIndex(0)
		{
			/* NO CODE */
		}

		/* CRTP OVERRIDE */ void NextCycle() noexcept
		{
			m_RobinIndex = (m_RobinIndex + 1) % Period();
		}

		/* CRTP OVERRIDE */ Int64 PairsPerCycle() const noexcept
		{
			return m_PairsPerCycle;
		}

		/* CRTP OVERRIDE */ Int64 Period() const noexcept
		{
			return m_Period;
		}

		/* CRTP OVERRIDE */ std::pair<Item, Item> operator[](const Int64 index) const
		{
			const auto i = index == 0 ? 0
			                          : (index > m_RobinIndex ? index - m_RobinIndex
			                                                  : m_EffectiveAndActiveRangeSize + index - m_RobinIndex);

			const auto j = [this, index]
			{
				if (m_EffectiveAndActiveRangeSize <= m_RobinIndex + index)
				{
					return 2 * m_EffectiveAndActiveRangeSize - (m_RobinIndex + index);
				}

				return m_EffectiveAndActiveRangeSize - (m_RobinIndex + index);
			}();

			return {std::move(ItemAt(i)), std::move(ItemAt(j))};
		}


	private:
		auto ItemAt(const Int64 index) const
		{
			if (index < m_RangeSize)
			{
				return m_Range[index];
			}
			else
			{
				return m_InvalidItemMarker;
			}
		}
	};


	template <typename Range>
	class BiPartiteRoundRobinOrdering : public AbstractRoundRobinOrdering<BiPartiteRoundRobinOrdering<Range>>
	{
	private:
		using Base = AbstractRoundRobinOrdering<BiPartiteRoundRobinOrdering<Range>>;
		using Item = std::decay_t<decltype(*std::declval<Range>().begin())>;
		Range m_Range0;
		Range m_Range1;
		Int64 m_RangeSize0;
		Int64 m_RangeSize1;
		Int64 m_PairsPerCycle;
		Int64 m_Period;
		Int64 m_RobinIndex;

	public:
		BiPartiteRoundRobinOrdering(Range range0, Range range1)
		    : m_Range0(std::move(range0)), m_Range1(std::move(range1)),
		      m_RangeSize0(std::distance(std::begin(m_Range0), std::end(m_Range0))),
		      m_RangeSize1(std::distance(std::begin(m_Range1), std::end(m_Range1))),
		      m_PairsPerCycle(std::min(m_RangeSize0, m_RangeSize1)), m_Period(std::max(m_RangeSize0, m_RangeSize1)),
		      m_RobinIndex(0)
		{
			/* NO CODE */
		}

		/* CRTP OVERRIDE */ void NextCycle() noexcept
		{
			m_RobinIndex = (m_RobinIndex + 1) % Period();
		}

		/* CRTP OVERRIDE */ Int64 PairsPerCycle() const noexcept
		{
			return m_PairsPerCycle;
		}

		/* CRTP OVERRIDE */ Int64 Period() const noexcept
		{
			return m_Period;
		}

		/* CRTP OVERRIDE */ std::pair<Item, Item> operator[](const Int64 index) const
		{
			if (m_RangeSize0 <= m_RangeSize1)
			{
				return {m_Range0[index], m_Range1[(index + m_RobinIndex) % m_RangeSize1]};
			}
			else
			{
				return {m_Range0[(index + m_RobinIndex) % m_RangeSize0], m_Range1[index]};
			}
		}
	};
}
