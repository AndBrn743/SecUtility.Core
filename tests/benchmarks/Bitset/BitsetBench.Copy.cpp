// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include "BitsetBench.Common.hpp"

namespace
{
	template <typename T, std::size_t N, bool Assign>
	void BM_Copy(benchmark::State& state)
	{
		auto original = RandomBitset<T, N>();
		auto dest = MakeBitset<T, N>();
		for (auto _ : state)
		{
			benchmark::DoNotOptimize(original);
			if constexpr (Assign)
			{
				dest = original;
				benchmark::DoNotOptimize(dest);
			}
			else
			{
				auto copy = original;
				benchmark::DoNotOptimize(copy);
			}
		}
	}

	template <typename T, std::size_t N>
	void RegisterOwner(const std::string& type)
	{
		const auto suffix = "/" + type + "/" + std::to_string(N);
		benchmark::RegisterBenchmark("CopyConstruct" + suffix, &BM_Copy<T, N, false>);
		benchmark::RegisterBenchmark("CopyAssign" + suffix, &BM_Copy<T, N, true>);
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

void BitsetBenchmark::RegisterCopies()
{
	RegisterSizes(OwnerSizes{});
}
