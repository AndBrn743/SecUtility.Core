// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include "BitsetBench.Common.hpp"

namespace
{
	enum class WholeSetOp
	{
		Set,
		Reset,
		Flip
	};

	template <typename T, std::size_t N, WholeSetOp Op>
	void BM_WholeSet(benchmark::State& state)
	{
		auto original = RandomBitset<T, N>();
		std::vector<T> work(BatchSize, original);
		for (auto _ : state)
		{
			ResetBatch(state, work, original);
			for (auto& bs : work)
			{
				if constexpr (IsStdBitset<T>::value)
				{
					if constexpr (Op == WholeSetOp::Set)
					{
						bs.set();
					}
					else if constexpr (Op == WholeSetOp::Reset)
					{
						bs.reset();
					}
					else
					{
						bs.flip();
					}
				}
				else
				{
					if constexpr (Op == WholeSetOp::Set)
					{
						bs.SetAll();
					}
					else if constexpr (Op == WholeSetOp::Reset)
					{
						bs.ResetAll();
					}
					else
					{
						bs.FlipAll();
					}
				}
				benchmark::DoNotOptimize(bs);
			}
		}
		state.SetItemsProcessed(state.iterations() * BatchSize);
	}

	template <typename T, std::size_t N>
	void RegisterOwner(const std::string& type)
	{
		const auto suffix = "/Batch32/" + type + "/" + std::to_string(N);
		benchmark::RegisterBenchmark("WholeSet/Set" + suffix, &BM_WholeSet<T, N, WholeSetOp::Set>);
		benchmark::RegisterBenchmark("WholeSet/Reset" + suffix, &BM_WholeSet<T, N, WholeSetOp::Reset>);
		benchmark::RegisterBenchmark("WholeSet/Flip" + suffix, &BM_WholeSet<T, N, WholeSetOp::Flip>);
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

void BitsetBenchmark::RegisterWholeSet()
{
	RegisterSizes(OwnerSizes{});
}
