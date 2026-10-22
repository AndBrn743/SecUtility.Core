// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

// Build, filters, and timing conventions are documented in BitsetBench.md.

#include <SecUtility/Collection/Bitset.hpp>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace SecUtility;

namespace
{
	constexpr std::size_t BatchSize = 32;
	constexpr std::size_t RuntimeShift = std::numeric_limits<std::size_t>::max();

	template <typename T>
	struct IsStdBitset : std::false_type
	{
	};
	template <std::size_t N>
	struct IsStdBitset<std::bitset<N>> : std::true_type
	{
	};

	template <typename T, std::size_t N>
	T MakeBitset()
	{
		if constexpr (std::is_same_v<T, DynamicBitset>)
		{
			return T(N);
		}
		else
		{
			return T{};
		}
	}

	template <typename T, std::size_t N>
	T RandomBitset(const std::uint64_t seed = 42)
	{
		auto bs = MakeBitset<T, N>();
		std::mt19937_64 rng(seed);
		for (std::size_t i = 0; i < N; ++i)
		{
			bs[i] = static_cast<bool>(rng() & 1u);
		}
		return bs;
	}

	template <bool Left, typename T>
	void Shift(T& bs, const std::size_t amount)
	{
		if constexpr (Left)
		{
			bs <<= amount;
		}
		else
		{
			bs >>= amount;
		}
	}

	// Reset existing allocations outside the measured interval. Batch timings
	// cover 32 operations; items_per_second reports individual bitsets per second.
	template <typename T>
	void ResetBatch(benchmark::State& state, std::vector<T>& work, T& original)
	{
		state.PauseTiming();
		benchmark::DoNotOptimize(original);
		std::fill(work.begin(), work.end(), original);
		benchmark::DoNotOptimize(work.data());
		state.ResumeTiming();
	}

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

	template <typename T, std::size_t N, bool Left, std::size_t Amount = RuntimeShift>
	void BM_CopyShift(benchmark::State& state)
	{
		auto original = RandomBitset<T, N>();
		const auto shift = [&]
		{
			if constexpr (Amount == RuntimeShift)
			{
				return static_cast<std::size_t>(state.range(0));
			}
			else
			{
				return Amount;
			}
		}();
		for (auto _ : state)
		{
			benchmark::DoNotOptimize(original);
			auto copy = original;
			Shift<Left>(copy, shift);
			benchmark::DoNotOptimize(copy);
		}
	}

	template <typename T, std::size_t N, bool Left>
	void BM_ShiftOnly(benchmark::State& state)
	{
		auto original = RandomBitset<T, N>();
		std::vector<T> work(BatchSize, original);
		const auto shift = static_cast<std::size_t>(state.range(0));
		for (auto _ : state)
		{
			ResetBatch(state, work, original);
			for (auto& bs : work)
			{
				Shift<Left>(bs, shift);
				benchmark::DoNotOptimize(bs);
			}
		}
		state.SetItemsProcessed(state.iterations() * BatchSize);
	}

	struct Assign
	{
		static constexpr const char* Name = "Assign";
		template <typename L, typename R>
		void operator()(L& lhs, const R& rhs) const
		{
			lhs = rhs;
		}
	};
	struct And
	{
		static constexpr const char* Name = "And";
		template <typename L, typename R>
		void operator()(L& lhs, const R& rhs) const
		{
			lhs &= rhs;
		}
	};
	struct Or
	{
		static constexpr const char* Name = "Or";
		template <typename L, typename R>
		void operator()(L& lhs, const R& rhs) const
		{
			lhs |= rhs;
		}
	};
	struct Xor
	{
		static constexpr const char* Name = "Xor";
		template <typename L, typename R>
		void operator()(L& lhs, const R& rhs) const
		{
			lhs ^= rhs;
		}
	};

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

	template <typename T>
	auto Evaluate(const T& expression)
	{
		if constexpr (IsStdBitset<T>::value)
		{
			return expression;
		}
		else
		{
			return expression.Eval();
		}
	}

	template <typename T, std::size_t N, bool Materialize>
	void BM_Expression(benchmark::State& state)
	{
		auto a = RandomBitset<T, N>();
		auto b = RandomBitset<T, N>(69);
		auto c = RandomBitset<T, N>(123);
		auto result = MakeBitset<T, N>();
		for (auto _ : state)
		{
			benchmark::DoNotOptimize(a);
			benchmark::DoNotOptimize(b);
			benchmark::DoNotOptimize(c);
			if constexpr (Materialize)
			{
				auto intermediate = Evaluate(a & b);
				// Require an actual intermediate store even when the compiler
				// could otherwise fuse the two eager standard-library operations.
				benchmark::DoNotOptimize(intermediate);
				result = intermediate | c;
			}
			else
			{
				result = (a & b) | c;
			}
			benchmark::DoNotOptimize(result);
		}
	}

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

	template <typename T, std::size_t N>
	void BM_Equal(benchmark::State& state, const int mismatch)
	{
		auto lhs = RandomBitset<T, N>();
		auto rhs = lhs;
		if constexpr (N != 0)
		{
			if (mismatch != 0)
			{
				const std::size_t index = mismatch == 1 ? 0 : N - 1;
				rhs[index] = !static_cast<bool>(rhs[index]);
			}
		}
		for (auto _ : state)
		{
			benchmark::DoNotOptimize(lhs);
			benchmark::DoNotOptimize(rhs);
			auto equal = lhs == rhs;
			benchmark::DoNotOptimize(equal);
		}
	}

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

	std::vector<std::size_t> ShiftAmounts(const std::size_t size)
	{
		std::vector<std::size_t> amounts{0, 1, 7, 63, 64, 65, 127, 256, size, size + 1};
		if (size != 0)
		{
			amounts.push_back(size - 1);
		}
		std::sort(amounts.begin(), amounts.end());
		amounts.erase(std::unique(amounts.begin(), amounts.end()), amounts.end());
		return amounts;
	}

	template <typename Benchmark>
	void AddShifts(Benchmark* bench, const std::size_t size)
	{
		bench->ArgName("shift");
		for (const auto shift : ShiftAmounts(size))
		{
			bench->Arg(static_cast<std::int64_t>(shift));
		}
	}

	template <typename T, std::size_t N>
	void RegisterOwner(const std::string& type)
	{
		const auto suffix = "/" + type + "/" + std::to_string(N);
		benchmark::RegisterBenchmark("CopyConstruct" + suffix, &BM_Copy<T, N, false>);
		benchmark::RegisterBenchmark("CopyAssign" + suffix, &BM_Copy<T, N, true>);
		AddShifts(benchmark::RegisterBenchmark("ShiftLeft/Copy" + suffix, &BM_CopyShift<T, N, true>), N);
		AddShifts(benchmark::RegisterBenchmark("ShiftRight/Copy" + suffix, &BM_CopyShift<T, N, false>), N);
		AddShifts(benchmark::RegisterBenchmark("ShiftLeft/Batch32" + suffix, &BM_ShiftOnly<T, N, true>), N);
		AddShifts(benchmark::RegisterBenchmark("ShiftRight/Batch32" + suffix, &BM_ShiftOnly<T, N, false>), N);
		if constexpr (N == 64 || N == 65 || N == 129 || N == 5409)
		{
			benchmark::RegisterBenchmark("ShiftLeft/Constant7" + suffix, &BM_CopyShift<T, N, true, 7>);
			benchmark::RegisterBenchmark("ShiftRight/Constant7" + suffix, &BM_CopyShift<T, N, false, 7>);
			benchmark::RegisterBenchmark("ShiftLeft/Constant64" + suffix, &BM_CopyShift<T, N, true, 64>);
			benchmark::RegisterBenchmark("ShiftRight/Constant64" + suffix, &BM_CopyShift<T, N, false, 64>);
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
	void RegisterOperations(const std::string& type)
	{
		const auto suffix = "/" + type + "/" + std::to_string(N);
		benchmark::RegisterBenchmark("Bitwise/And/Batch32" + suffix, &BM_Compound<T, N, And>);
		benchmark::RegisterBenchmark("Bitwise/Or/Batch32" + suffix, &BM_Compound<T, N, Or>);
		benchmark::RegisterBenchmark("Bitwise/Xor/Batch32" + suffix, &BM_Compound<T, N, Xor>);
		benchmark::RegisterBenchmark("Expression/Chained" + suffix, &BM_Expression<T, N, false>);
		benchmark::RegisterBenchmark("Expression/Materialized" + suffix, &BM_Expression<T, N, true>);
		RegisterQuery<T, N, Query::Count>("Count", suffix);
		RegisterQuery<T, N, Query::Any>("Any", suffix);
		// Empty all/none differ between the APIs, so do not compare them.
		if constexpr (N != 0)
		{
			RegisterQuery<T, N, Query::All>("All", suffix);
			RegisterQuery<T, N, Query::None>("None", suffix);
		}
		benchmark::RegisterBenchmark("Equal" + suffix + "/Same", &BM_Equal<T, N>, 0);
		if constexpr (N != 0)
		{
			benchmark::RegisterBenchmark("Equal" + suffix + "/FirstMismatch", &BM_Equal<T, N>, 1);
			benchmark::RegisterBenchmark("Equal" + suffix + "/LastMismatch", &BM_Equal<T, N>, 2);
		}
	}

	template <std::size_t N>
	void RegisterSize()
	{
		RegisterOwner<Bitset<N>, N>("Static");
		RegisterOwner<DynamicBitset, N>("Dynamic");
		RegisterOwner<std::bitset<N>, N>("Std");
		if constexpr (N == 0 || N == 64 || N == 65 || N == 129 || N == 5409)
		{
			RegisterOperations<Bitset<N>, N>("Static");
			RegisterOperations<DynamicBitset, N>("Dynamic");
			RegisterOperations<std::bitset<N>, N>("Std");
		}
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
}

int main(int argc, char** argv)
{
	RegisterSize<0>();
	RegisterSize<1>();
	RegisterSize<42>();
	RegisterSize<63>();
	RegisterSize<64>();
	RegisterSize<65>();
	RegisterSize<127>();
	RegisterSize<128>();
	RegisterSize<129>();
	RegisterSize<549>();
	RegisterSize<5409>();
	RegisterSize<65536>();
	RegisterSegments<42>();
	RegisterSegments<65>();
	RegisterSegments<129>();
	RegisterSegments<5409>();
	benchmark::Initialize(&argc, argv);
	if (benchmark::ReportUnrecognizedArguments(argc, argv))
	{
		return 1;
	}
	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();
}
