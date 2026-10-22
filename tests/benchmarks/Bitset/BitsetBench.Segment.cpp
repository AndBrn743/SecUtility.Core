// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include "BitsetBench.Common.hpp"

namespace
{
	// Views have no std::bitset counterpart. Source and destination use separate
	// allocations; SrcView=false also exercises owning-to-view realignment.
	template <typename Source, std::size_t N, typename Op, bool SrcView>
	void BM_SegmentBinary(benchmark::State& state)
	{
		auto sourceStorage = RandomBitset<Source, SrcView ? N + 64 : N>();
		auto original = RandomBitset<DynamicBitset, N + 64>(69);
		std::vector<DynamicBitset> work(BatchSize, original);
		const auto sourceHead = static_cast<std::size_t>(state.range(0));
		const auto destHead = static_cast<std::size_t>(state.range(1));
		decltype(auto) source = [&]() -> decltype(auto)
		{
			if constexpr (SrcView)
			{
				return std::as_const(sourceStorage).Segment(sourceHead, N);
			}
			else
			{
				return std::as_const(sourceStorage);
			}
		}();
		for (auto _ : state)
		{
			ResetBatch(state, work, original);
			benchmark::DoNotOptimize(sourceStorage);
			for (auto& bs : work)
			{
				auto dest = bs.Segment(destHead, N);
				Op{}(dest, source);
				benchmark::DoNotOptimize(bs);
			}
		}
		state.SetItemsProcessed(state.iterations() * BatchSize);
	}

	template <std::size_t N, bool Left>
	void BM_SegmentShift(benchmark::State& state)
	{
		auto original = RandomBitset<DynamicBitset, N + 64>();
		std::vector<DynamicBitset> work(BatchSize, original);
		const auto head = static_cast<std::size_t>(state.range(0));
		const auto shift = static_cast<std::size_t>(state.range(1));
		for (auto _ : state)
		{
			ResetBatch(state, work, original);
			for (auto& bs : work)
			{
				auto segment = bs.Segment(head, N);
				Shift<Left>(segment, shift);
				benchmark::DoNotOptimize(bs);
			}
		}
		state.SetItemsProcessed(state.iterations() * BatchSize);
	}

	template <std::size_t N, typename Op>
	void RegisterSegmentBinary()
	{
		const auto name = std::string("Segment/") + Op::Name + "/Batch32/";
		auto* views = benchmark::RegisterBenchmark(name + "Views/" + std::to_string(N),
		                                           &BM_SegmentBinary<DynamicBitset, N, Op, true>);
		views->ArgNames({"source_head", "dest_head"});
		for (const auto& heads : {std::pair<int, int>{0, 0}, {11, 11}, {0, 19}, {19, 0}, {11, 42}, {42, 11}})
		{
			views->Args({heads.first, heads.second});
		}
		auto* fixed = benchmark::RegisterBenchmark(name + "StaticSource/" + std::to_string(N),
		                                           &BM_SegmentBinary<Bitset<N>, N, Op, false>);
		auto* dynamic = benchmark::RegisterBenchmark(name + "DynamicSource/" + std::to_string(N),
		                                             &BM_SegmentBinary<DynamicBitset, N, Op, false>);
		for (auto* bench : {fixed, dynamic})
		{
			bench->ArgNames({"source_head", "dest_head"});
			for (const auto head : {0, 19, 63})
			{
				bench->Args({0, head});
			}
		}
	}

	template <std::size_t N>
	void RegisterSegments()
	{
		RegisterSegmentBinary<N, Assign>();
		RegisterSegmentBinary<N, And>();
		RegisterSegmentBinary<N, Or>();
		RegisterSegmentBinary<N, Xor>();
		auto* left = benchmark::RegisterBenchmark("Segment/ShiftLeft/Batch32/" + std::to_string(N),
		                                          &BM_SegmentShift<N, true>);
		auto* right = benchmark::RegisterBenchmark("Segment/ShiftRight/Batch32/" + std::to_string(N),
		                                           &BM_SegmentShift<N, false>);
		for (auto* bench : {left, right})
		{
			bench->ArgNames({"head", "shift"});
			for (const auto head : {0, 11, 63})
			{
				for (const auto shift : ShiftAmounts(N))
				{
					bench->Args({head, static_cast<std::int64_t>(shift)});
				}
			}
		}
	}

	template <std::size_t... Sizes>
	void RegisterSizes(std::index_sequence<Sizes...>)
	{
		(RegisterSegments<Sizes>(), ...);
	}
}

void BitsetBenchmark::RegisterSegments()
{
	RegisterSizes(SegmentSizes{});
}
