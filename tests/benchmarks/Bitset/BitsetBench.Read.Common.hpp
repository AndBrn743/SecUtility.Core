// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include "BitsetBench.Common.hpp"

namespace
{
	enum class ReadForm
	{
		Owner,
		Not,
		And,
		Nested,
		NotNested,
		View,
		NotView,
		ViewAnd,
		ViewNested
	};

	enum class ReadPattern
	{
		Random,
		Zero,
		Ones,
		FirstOne,
		MiddleOne,
		LastOne,
		Sparse,
		SparseZeros
	};

	constexpr std::pair<ReadPattern, const char*> ReadPatterns[] = {{ReadPattern::Random, "Random"},
	                                                                {ReadPattern::Zero, "Zero"},
	                                                                {ReadPattern::Ones, "Ones"},
	                                                                {ReadPattern::FirstOne, "FirstOne"},
	                                                                {ReadPattern::MiddleOne, "MiddleOne"},
	                                                                {ReadPattern::LastOne, "LastOne"},
	                                                                {ReadPattern::Sparse, "Sparse"},
	                                                                {ReadPattern::SparseZeros, "SparseZeros"}};

	constexpr const char* ReadFormName(const ReadForm form)
	{
		switch (form)
		{
			case ReadForm::Owner:
				return "Owner";
			case ReadForm::Not:
				return "Not";
			case ReadForm::And:
				return "And";
			case ReadForm::Nested:
				return "Nested";
			case ReadForm::NotNested:
				return "NotNested";
			case ReadForm::View:
				return "View";
			case ReadForm::NotView:
				return "NotView";
			case ReadForm::ViewAnd:
				return "ViewAnd";
			case ReadForm::ViewNested:
				return "ViewNested";
		}
		return "";
	}

	template <ReadForm Form>
	constexpr bool UsesReadView = Form == ReadForm::View || Form == ReadForm::NotView || Form == ReadForm::ViewAnd
	                              || Form == ReadForm::ViewNested;

	inline bool ReadPatternBit(const ReadPattern pattern,
	                           const std::size_t i,
	                           const std::size_t size,
	                           std::mt19937_64& rng)
	{
		switch (pattern)
		{
			case ReadPattern::Random:
				return (rng() & 1u) != 0;
			case ReadPattern::Zero:
				return false;
			case ReadPattern::Ones:
				return true;
			case ReadPattern::FirstOne:
				return i == 0;
			case ReadPattern::MiddleOne:
				return i == size / 2;
			case ReadPattern::LastOne:
				return i + 1 == size;
			case ReadPattern::Sparse:
				return i % 257 == (size / 2) % 257;
			case ReadPattern::SparseZeros:
				return i % 257 != (size / 2) % 257;
		}
		return false;
	}

	// Patterns describe the logical result, independent of the expression form.
	// Runtime-barriered all-one/all-zero operands keep AND/nested results at the
	// requested density without allowing the compiler to simplify those operands.
	template <typename T, std::size_t N, ReadForm Form>
	struct ReadFixture
	{
		using Backing = std::conditional_t<std::is_same_v<T, DynamicBitset>, DynamicBitset, Bitset<N + 128>>;

		T a = MakeBitset<T, N>();
		T b = MakeBitset<T, N>();
		T c = MakeBitset<T, N>();
		Backing backing = MakeBitset<Backing, N + 128>();
		std::bitset<N> expected;
		std::size_t head;

		ReadFixture(const ReadPattern pattern, const std::size_t head) : head(head)
		{
			if constexpr (UsesReadView<Form>)
			{
				// Guard bits outside the segment expose reads beyond its payload.
				backing.SetAll();
			}
			std::mt19937_64 rng(42);
			for (std::size_t i = 0; i < N; ++i)
			{
				const bool value = ReadPatternBit(pattern, i, N, rng);
				expected[i] = value;
				a[i] = Form == ReadForm::Not || Form == ReadForm::NotNested
				               ? !value
				               : Form == ReadForm::ViewAnd || Form == ReadForm::ViewNested || value;
				b[i] = Form != ReadForm::ViewNested;
				c[i] = false;
				if constexpr (UsesReadView<Form>)
				{
					backing[64 + head + i] = Form == ReadForm::NotView ? !value : value;
				}
			}
		}

		void Barrier()
		{
			if constexpr (UsesReadView<Form>)
			{
				benchmark::DoNotOptimize(backing);
			}
			if constexpr (!UsesReadView<Form> || Form == ReadForm::ViewAnd || Form == ReadForm::ViewNested)
			{
				benchmark::DoNotOptimize(a);
			}
			if constexpr (Form == ReadForm::And || Form == ReadForm::Nested || Form == ReadForm::NotNested
			              || Form == ReadForm::ViewNested)
			{
				benchmark::DoNotOptimize(b);
			}
			if constexpr (Form == ReadForm::Nested || Form == ReadForm::NotNested)
			{
				benchmark::DoNotOptimize(c);
			}
		}

		template <typename Function>
		decltype(auto) Apply(Function&& function) const
		{
			if constexpr (Form == ReadForm::Owner)
			{
				return function(a);
			}
			else if constexpr (Form == ReadForm::Not)
			{
				return function(~a);
			}
			else if constexpr (Form == ReadForm::And)
			{
				return function(a & b);
			}
			else if constexpr (Form == ReadForm::Nested)
			{
				return function((a & b) | c);
			}
			else if constexpr (Form == ReadForm::NotNested)
			{
				return function(~((a & b) | c));
			}
			else if constexpr (Form == ReadForm::View)
			{
				return function(backing.Segment(64 + head, N));
			}
			else if constexpr (Form == ReadForm::NotView)
			{
				return function(~backing.Segment(64 + head, N));
			}
			else
			{
				if constexpr (Form == ReadForm::ViewNested)
				{
					return function((backing.Segment(64 + head, N) & a) | b);
				}
				else
				{
					return function(backing.Segment(64 + head, N) & a);
				}
			}
		}
	};

	template <ReadForm Form>
	std::string ReadSuffix(const std::string& type, const std::size_t size, const std::size_t head)
	{
		auto suffix = std::string("/") + ReadFormName(Form) + "/" + type + "/" + std::to_string(size);
		if constexpr (UsesReadView<Form>)
		{
			suffix += "/Head" + std::to_string(head);
		}
		return suffix;
	}

	inline bool CheckReadResult(benchmark::State& state, const std::size_t actual, const std::size_t expected)
	{
		if (actual != expected)
		{
			state.SkipWithError("Read result disagrees with the indexed reference");
			return false;
		}
		return true;
	}
}
