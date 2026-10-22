// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include "BitsetBench.Common.hpp"

namespace
{
	enum class Pattern
	{
		Random,
		Zero,
		Ones,
		First,
		Last
	};
	enum class Query
	{
		Count,
		All,
		Any,
		None
	};

	template <typename T, std::size_t N, Query Q>
	void BM_Query(benchmark::State& state, const Pattern pattern)
	{
		auto bs = RandomBitset<T, N>();
		if (pattern != Pattern::Random)
		{
			for (std::size_t i = 0; i < N; ++i)
			{
				bs[i] = pattern == Pattern::Ones || (pattern == Pattern::First && i == 0)
				        || (pattern == Pattern::Last && i + 1 == N);
			}
		}
		for (auto _ : state)
		{
			benchmark::DoNotOptimize(bs);
			auto result = [&]
			{
				if constexpr (IsStdBitset<T>::value)
				{
					if constexpr (Q == Query::Count)
					{
						return bs.count();
					}
					else if constexpr (Q == Query::All)
					{
						return bs.all();
					}
					else if constexpr (Q == Query::Any)
					{
						return bs.any();
					}
					else
					{
						return bs.none();
					}
				}
				else
				{
					if constexpr (Q == Query::Count)
					{
						return bs.OneCount();
					}
					else if constexpr (Q == Query::All)
					{
						return bs.IsAllOnes();
					}
					else if constexpr (Q == Query::Any)
					{
						return bs.HasOnes();
					}
					else
					{
						return bs.IsAllZeros();
					}
				}
			}();
			benchmark::DoNotOptimize(result);
		}
	}

	template <typename T, std::size_t N, Query Q>
	void RegisterQuery(const std::string& operation, const std::string& suffix)
	{
		const std::pair<Pattern, const char*> patterns[] = {{Pattern::Random, "Random"},
		                                                    {Pattern::Zero, "Zero"},
		                                                    {Pattern::Ones, "Ones"},
		                                                    {Pattern::First, "FirstOne"},
		                                                    {Pattern::Last, "LastOne"}};
		for (const auto& pattern : patterns)
		{
			if (N == 0 && pattern.first != Pattern::Random)
			{
				continue;
			}
			benchmark::RegisterBenchmark(
			        "Query/" + operation + suffix + "/" + pattern.second, &BM_Query<T, N, Q>, pattern.first);
		}
	}

	template <typename T, std::size_t N>
	void RegisterOwner(const std::string& type)
	{
		const auto suffix = "/" + type + "/" + std::to_string(N);
		RegisterQuery<T, N, Query::Count>("Count", suffix);
		RegisterQuery<T, N, Query::Any>("Any", suffix);
		// Empty all/none differ between the APIs, so do not compare them.
		if constexpr (N != 0)
		{
			RegisterQuery<T, N, Query::All>("All", suffix);
			RegisterQuery<T, N, Query::None>("None", suffix);
		}
	}

	template <std::size_t N>
	void RegisterSize()
	{
		RegisterOwner<Bitset<N>, N>("Static");
		RegisterOwner<DynamicBitset, N>("Dynamic");
		RegisterOwner<std::bitset<N>, N>("Std");
	}

	template <std::size_t... Sizes>
	void RegisterSizes(std::index_sequence<Sizes...>)
	{
		(RegisterSize<Sizes>(), ...);
	}
}

void BitsetBenchmark::RegisterQueries()
{
	RegisterSizes(OperationSizes{});
}
