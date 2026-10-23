// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

// Build, filters, and timing conventions are documented in BitsetBench.md.

#include "BitsetBench.Common.hpp"

int main(int argc, char** argv)
{
	BitsetBenchmark::RegisterCopies();
	BitsetBenchmark::RegisterShifts();
	BitsetBenchmark::RegisterBitwise();
	BitsetBenchmark::RegisterWholeSet();
	BitsetBenchmark::RegisterExpressions();
	BitsetBenchmark::RegisterQueries();
	BitsetBenchmark::RegisterEquality();
	BitsetBenchmark::RegisterSegments();
	BitsetBenchmark::RegisterExpressionReads();
	BitsetBenchmark::RegisterScans();
	benchmark::Initialize(&argc, argv);
	if (benchmark::ReportUnrecognizedArguments(argc, argv))
	{
		return 1;
	}
	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();
}
