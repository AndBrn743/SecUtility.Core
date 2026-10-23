# Bitset benchmarks

Build with Google Benchmark installed:

```sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DSECUTILITY_BUILD_BENCHMARKS=ON
cmake --build build-bench --target BitsetBench
./build-bench/tests/BitsetBench --benchmark_list_tests=true
```

For instruction-set comparisons, use the same compiler flags for every build.
The target uses the project's normal Release flags; it does not enable
`-march=native` automatically.

Benchmark families compile separately: `BitsetBench.Copy.cpp`,
`BitsetBench.Shift.cpp`, `BitsetBench.Bitwise.cpp`, `BitsetBench.Expression.cpp`,
`BitsetBench.WholeSet.cpp`, `BitsetBench.Query.cpp`, `BitsetBench.Equality.cpp`,
and `BitsetBench.Segment.cpp`.
`BitsetBench.cpp` explicitly registers each family and runs Google Benchmark.
`BitsetBench.Common.hpp` shares fixtures, operation adapters, and size matrices;
its helpers have internal linkage so families can optimize independently.

The target disables CMake unity builds and interprocedural optimization to
preserve these boundaries. Avoid adding LTO flags manually when comparing
families. Case names, arguments, input seeds, timing barriers, and batch sizes
are unchanged; registration is now grouped by family. Compiler optimization
decisions can change after this split, so establish new timing baselines before
comparing subsequent library changes.

The expanded suite uses names containing the operation, measurement mode,
storage type (`Static`, `Dynamic`, or `Std`), and bit count. These replace the
earlier `BM_*` names. Use filters to select a manageable comparison:

```sh
# Runtime shifts around word boundaries, including the former 5,409-bit case.
./build-bench/tests/BitsetBench \
  --benchmark_filter='^Shift(Left|Right)/(Copy|Batch32)/(Static|Std)/(64|65|129|5409)/' \
  --benchmark_min_time=0.05s \
  --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
  --benchmark_out=bitset-shifts.json --benchmark_out_format=json

# Counts, predicates, and equality, including early and late exits.
./build-bench/tests/BitsetBench --benchmark_filter='^(Query|Equal)/'

# Whole-set and compound operations on preallocated storage.
./build-bench/tests/BitsetBench --benchmark_filter='^(WholeSet|Bitwise)/'

# Alignment and source/destination block-count differences.
./build-bench/tests/BitsetBench --benchmark_filter='^Segment/'

# Smoke test every registered case; these timings are not performance results.
./build-bench/tests/BitsetBench --benchmark_min_time=0.0001s
```

## Measurements

| Family | Timed work |
|---|---|
| `CopyConstruct` | Construct and destroy a copy; dynamic allocation is included. |
| `CopyAssign` | Assign into existing storage of the same size. |
| `ShiftLeft/Copy`, `ShiftRight/Copy` | Construct a copy, shift by a runtime amount, and destroy the copy. |
| `ShiftLeft/Constant7`, `ShiftLeft/Constant64` and right-shift counterparts | Copy plus a compile-time constant shift. |
| `ShiftLeft/Batch32`, `ShiftRight/Batch32` | Shift 32 preallocated bitsets, each starting with the same random pattern. |
| `Bitwise/*/Batch32` | Apply `&=`, `|=`, or `^=` to 32 preallocated bitsets. |
| `WholeSet/*/Batch32` | Set, reset, or flip all bits in 32 preallocated bitsets. |
| `Expression/Chained` | Assign `(a & b) | c` into an existing destination. |
| `Expression/Materialized` | Materialize `a & b`, then assign its OR with `c`; intermediate allocation and destruction are included. |
| `Expression/AndAll` | Query `(a & b).IsAllOnes()` or `(a & b).all()`, with random inputs, all ones, or a zero at the final bit. |
| `Query` | One count or predicate on the named input pattern. |
| `Equal` | One equality comparison, with identical inputs or a mismatch at bit 0 or bit N-1. |
| `Segment/*/Batch32` | Apply a shift or binary operation to 32 segments in existing backing storage. |

**Batch32 timings are per batch, not per individual operation.** Divide the
reported time by 32 to obtain time per bitset, or compare `items_per_second`,
which already counts individual bitsets. Each iteration restores all 32 inputs
with timing paused, then resumes timing for the operations. This prevents
repeated shifts or AND operations from degenerating to all-zero inputs and
keeps reset copying and allocation outside the measured interval. Loop overhead,
memory access across the batch, and segment construction remain included.
Batch results therefore describe a different workload from single-copy cases.

All inputs use deterministic seeds. Random generation occurs before timing;
input and output barriers prevent hoisting or elimination. Query patterns are
random, all zero, all one, only bit 0 set, and only bit N-1 set. Empty all/none
queries are omitted because the two APIs give different answers for those cases.

Chained expressions use each library's normal evaluation rules. The materialized
variant additionally places an optimization barrier on the intermediate so that
the compiler must store it, including for `std::bitset`.

`Expression/AndAll` also uses normal evaluation rules, with no intermediate
barrier. Random inputs exercise early failure; all-ones and last-zero inputs
require a full scan of the lazy expression. Empty queries are omitted because
the APIs differ. Repeated inputs give predictable branch outcomes.

## Coverage

- Copying, whole-set operations, and runtime shifts: 0, 1, 42, 63, 64, 65, 127,
  128, 129, 549, 5,409, and 65,536 bits, for static, dynamic, and standard bitsets.
- Runtime shift amounts: deduplicated values from 0, 1, 7, 63, 64, 65, 127, 256,
  N-1, N, and N+1. N-1 is omitted when N is zero.
- Constant shifts: 7 and 64, at sizes 64, 65, 129, and 5,409.
- Bitwise operations, expressions, counts, predicates, and equality: sizes 0,
  64, 65, 129, and 5,409, with empty cases restricted where appropriate.
- Segments: sizes 42, 65, 129, and 5,409. Shifts use head offsets 0, 11, and 63.
  Binary operations cover matching and differing head offsets, in both alignment
  directions, plus static and dynamic owning sources assigned or combined into
  views. Source and destination allocations are distinct; overlap is not measured.

The larger operation matrix uses representative sizes rather than crossing
every size with every input pattern. The full suite can take a long time with
Google Benchmark's default duration; use family and size filters during tuning.
