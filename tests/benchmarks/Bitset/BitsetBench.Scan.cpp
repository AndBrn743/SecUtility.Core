// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#include "BitsetBench.Read.Common.hpp"

namespace
{
	enum class ScanQuery
	{
		TrailingZeros,
		LeadingZeros,
		TrailingOnes,
		LeadingOnes,
		FirstOne,
		LastOne,
		FirstZero,
		LastZero,
		NextOne,
		NextZero,
		PreviousOne,
		PreviousZero
	};

	template <ScanQuery Query>
	constexpr bool IsCursorScan = Query >= ScanQuery::NextOne;

	// Portable indexed reference. std::bitset has no native scan API, so its
	// timed baseline is explicitly named StdIndexed rather than Std.
	template <ScanQuery Query, std::size_t N>
	std::size_t IndexedScan(const std::bitset<N>& bits, const std::size_t pos)
	{
		if constexpr (Query == ScanQuery::TrailingZeros || Query == ScanQuery::TrailingOnes
		              || Query == ScanQuery::LeadingZeros || Query == ScanQuery::LeadingOnes)
		{
			constexpr bool ones = Query == ScanQuery::TrailingOnes || Query == ScanQuery::LeadingOnes;
			constexpr bool reverse = Query == ScanQuery::LeadingZeros || Query == ScanQuery::LeadingOnes;
			std::size_t count = 0;
			for (std::size_t i = 0; i < N; ++i)
			{
				if (bits[reverse ? N - 1 - i : i] != ones)
				{
					break;
				}
				++count;
			}
			return count;
		}
		else
		{
			constexpr bool ones = Query == ScanQuery::FirstOne || Query == ScanQuery::LastOne
			                      || Query == ScanQuery::NextOne || Query == ScanQuery::PreviousOne;
			if constexpr (Query == ScanQuery::FirstOne || Query == ScanQuery::FirstZero || Query == ScanQuery::NextOne
			              || Query == ScanQuery::NextZero)
			{
				const std::size_t first = IsCursorScan<Query> ? pos + 1 : 0;
				for (std::size_t i = first; i < N; ++i)
				{
					if (bits[i] == ones)
					{
						return i;
					}
				}
			}
			else
			{
				for (std::size_t i = IsCursorScan<Query> ? pos : N; i-- > 0;)
				{
					if (bits[i] == ones)
					{
						return i;
					}
				}
			}
			return N;
		}
	}

	template <ScanQuery Query, typename Expression>
	std::size_t QueryScan(const Expression& expression, const std::size_t pos)
	{
		if constexpr (IsStdBitset<Expression>::value)
		{
			return IndexedScan<Query>(expression, pos);
		}
		else
		{
			if constexpr (Query == ScanQuery::TrailingZeros)
			{
				return expression.TrailingZeroCount();
			}
			else if constexpr (Query == ScanQuery::LeadingZeros)
			{
				return expression.LeadingZeroCount();
			}
			else if constexpr (Query == ScanQuery::TrailingOnes)
			{
				return expression.TrailingOneCount();
			}
			else if constexpr (Query == ScanQuery::LeadingOnes)
			{
				return expression.LeadingOneCount();
			}
			else if constexpr (Query == ScanQuery::FirstOne)
			{
				return expression.IndexOfFirstOne();
			}
			else if constexpr (Query == ScanQuery::LastOne)
			{
				return expression.IndexOfLastOne();
			}
			else if constexpr (Query == ScanQuery::FirstZero)
			{
				return expression.IndexOfFirstZero();
			}
			else if constexpr (Query == ScanQuery::LastZero)
			{
				return expression.IndexOfLastZero();
			}
			else if constexpr (Query == ScanQuery::NextOne)
			{
				return expression.IndexOfNextOne(pos);
			}
			else if constexpr (Query == ScanQuery::NextZero)
			{
				return expression.IndexOfNextZero(pos);
			}
			else if constexpr (Query == ScanQuery::PreviousOne)
			{
				return expression.IndexOfPreviousOne(pos);
			}
			else
			{
				return expression.IndexOfPreviousZero(pos);
			}
		}
	}

	template <typename T, std::size_t N, ReadForm Form, ScanQuery Query>
	void BM_Scan(benchmark::State& state, const ReadPattern pattern, const std::size_t head)
	{
		ReadFixture<T, N, Form> fixture(pattern, head);
		const std::size_t pos = IsCursorScan<Query> ? static_cast<std::size_t>(state.range(0)) : 0;
		const auto query = [pos](const auto& expression) { return QueryScan<Query>(expression, pos); };
		if (!CheckReadResult(state, fixture.Apply(query), IndexedScan<Query>(fixture.expected, pos)))
		{
			return;
		}
		for (auto _ : state)
		{
			fixture.Barrier();
			auto result = fixture.Apply(query);
			benchmark::DoNotOptimize(result);
		}
	}

	template <ScanQuery Query, std::size_t N>
	std::vector<std::size_t> ScanCursors()
	{
		std::vector<std::size_t> positions{0, N / 2, N == 0 ? 0 : N - 1};
		if constexpr (N > 64)
		{
			positions.push_back(63);
			positions.push_back(64);
		}
		if constexpr (Query == ScanQuery::PreviousOne || Query == ScanQuery::PreviousZero)
		{
			positions.push_back(N);
		}
		std::sort(positions.begin(), positions.end());
		positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
		return positions;
	}

	template <typename T, std::size_t N, ReadForm Form, ScanQuery Query>
	void RegisterScanQuery(const std::string& name, const std::string& suffix, const std::size_t head)
	{
		for (const auto& pattern : ReadPatterns)
		{
			if (N == 0 && pattern.first != ReadPattern::Random)
			{
				continue;
			}
			auto* bench = benchmark::RegisterBenchmark(
			        "Scan/" + name + suffix + "/" + pattern.second, &BM_Scan<T, N, Form, Query>, pattern.first, head);
			if constexpr (IsCursorScan<Query>)
			{
				bench->ArgName("pos");
				for (const auto pos : ScanCursors<Query, N>())
				{
					bench->Arg(static_cast<std::int64_t>(pos));
				}
			}
		}
	}

	template <typename T, std::size_t N, ReadForm Form>
	void RegisterScanForm(const std::string& type, const std::size_t head = 0)
	{
		const auto suffix = ReadSuffix<Form>(type, N, head);
		RegisterScanQuery<T, N, Form, ScanQuery::TrailingZeros>("TrailingZeros", suffix, head);
		RegisterScanQuery<T, N, Form, ScanQuery::LeadingZeros>("LeadingZeros", suffix, head);
		RegisterScanQuery<T, N, Form, ScanQuery::TrailingOnes>("TrailingOnes", suffix, head);
		RegisterScanQuery<T, N, Form, ScanQuery::LeadingOnes>("LeadingOnes", suffix, head);
		RegisterScanQuery<T, N, Form, ScanQuery::FirstOne>("FirstOne", suffix, head);
		RegisterScanQuery<T, N, Form, ScanQuery::LastOne>("LastOne", suffix, head);
		RegisterScanQuery<T, N, Form, ScanQuery::FirstZero>("FirstZero", suffix, head);
		RegisterScanQuery<T, N, Form, ScanQuery::LastZero>("LastZero", suffix, head);
		RegisterScanQuery<T, N, Form, ScanQuery::NextOne>("NextOne", suffix, head);
		RegisterScanQuery<T, N, Form, ScanQuery::NextZero>("NextZero", suffix, head);
		RegisterScanQuery<T, N, Form, ScanQuery::PreviousOne>("PreviousOne", suffix, head);
		RegisterScanQuery<T, N, Form, ScanQuery::PreviousZero>("PreviousZero", suffix, head);
	}

	template <typename T, std::size_t N>
	void RegisterScanOwner(const std::string& type)
	{
		RegisterScanForm<T, N, ReadForm::Owner>(type);
		RegisterScanForm<T, N, ReadForm::Not>(type);
		RegisterScanForm<T, N, ReadForm::And>(type);
		RegisterScanForm<T, N, ReadForm::Nested>(type);
		RegisterScanForm<T, N, ReadForm::NotNested>(type);
		for (const std::size_t head : {0u, 11u, 63u})
		{
			RegisterScanForm<T, N, ReadForm::View>(type, head);
			RegisterScanForm<T, N, ReadForm::NotView>(type, head);
			RegisterScanForm<T, N, ReadForm::ViewAnd>(type, head);
		}
	}

	template <std::size_t... Sizes>
	void RegisterScanSizes(std::index_sequence<Sizes...>)
	{
		(RegisterScanOwner<Bitset<Sizes>, Sizes>("Static"), ...);
		(RegisterScanOwner<DynamicBitset, Sizes>("Dynamic"), ...);
		(RegisterScanForm<std::bitset<Sizes>, Sizes, ReadForm::Owner>("StdIndexed"), ...);
	}
}

void BitsetBenchmark::RegisterScans()
{
	RegisterScanSizes(OperationSizes{});
}
