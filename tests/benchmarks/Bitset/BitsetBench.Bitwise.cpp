// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include "BitsetBench.Common.hpp"

namespace
{
	template <typename T, std::size_t N, typename Op>
	void BM_Compound(benchmark::State& state)
	{
		auto original = RandomBitset<T, N>();
		auto rhs = RandomBitset<T, N>(69);
		std::vector<T> work(BatchSize, original);
		for (auto _ : state)
		{
			ResetBatch(state, work, original);
			benchmark::DoNotOptimize(rhs);
			for (auto& bs : work)
			{
				Op{}(bs, rhs);
				benchmark::DoNotOptimize(bs);
			}
		}
		state.SetItemsProcessed(state.iterations() * BatchSize);
	}

	template <typename T, std::size_t N>
	void RegisterOwner(const std::string& type)
	{
		const auto suffix = "/" + type + "/" + std::to_string(N);
		benchmark::RegisterBenchmark("Bitwise/And/Batch32" + suffix, &BM_Compound<T, N, And>);
		benchmark::RegisterBenchmark("Bitwise/Or/Batch32" + suffix, &BM_Compound<T, N, Or>);
		benchmark::RegisterBenchmark("Bitwise/Xor/Batch32" + suffix, &BM_Compound<T, N, Xor>);
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

void BitsetBenchmark::RegisterBitwise()
{
	RegisterSizes(OperationSizes{});
}
