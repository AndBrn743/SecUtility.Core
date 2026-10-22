# Iterative vector interaction interior eigensolver

`IterativeVectorInteractionSelfAdjointEigenSolver` finds all eigenpairs of a
real symmetric or complex Hermitian linear operator in an inclusive interval.
It is matrix-free: the operator supplies its dimensions, real diagonal, and an
`ApplyOn` operation for vectors or blocks of vectors. The implementation and
its header require C++20; the rest of SecUtility.Core retains its lower language
requirement.

The solver returns an algorithmic terminal `InteriorEigenSolverStatus`; it does
not have a separate no-eigenpairs status. Inspect `Eigenvalues().empty()` when
distinguishing an empty interval from other terminal outcomes. For example, an
exact retained space with no interval eigenvalues terminates as
`ExpansionSpaceExhausted` with empty result containers. Non-converged runs retain
their latest interval Ritz approximations for diagnosis. Returned eigenvalues
are ascending, and eigenvector column and residual entry `i` belong to
eigenvalue `i`.

## Subspace extensions

`IterativeVectorInteractionSubspaceExtension` is a bitmask. Its descriptive
names map to the buffers in the iVI paper as follows:

| Flag | Paper name | Direction added to the next expansion space | Principal cost |
|---|---:|---|---|
| `AdditionalRitzVectors` | buffer 0 | Unconverged interval Ritz vectors and selected nearby Ritz vectors | Larger retained subspace |
| `CorrectionVectorImages` | buffer 1 | Images of accepted ordinary correction directions | One extra operator image per retained correction direction |
| `PreconditionedOffDiagonalCorrectionImages` | buffer 2 | Diagonally preconditioned projected `(H - D)q` directions | One extra operator image per retained auxiliary direction, plus preconditioning |
| `PreviousRitzVectors` | buffer 3 | Matched Ritz directions recycled from the preceding iteration | Larger retained subspace and matching work |

The ordinary correction direction is always generated; these flags control the
additional directions retained around it. Any combination of the four flags is
valid, including `None` and `All`. Candidates from the two image extensions are
concatenated and rank-filtered together, so a direction dependent across the
two sources is inserted only once.

The default is `AdditionalRitzVectors | PreviousRitzVectors`, corresponding to
paper buffers 0+3 and preserving the v1 behavior. Settings belonging to a
disabled extension are ignored. `IsPreviousRitzVectorRecyclingDynamicallyEnabled`
allows buffer 3 to stop recycling after refinement has reached its configured
threshold; it does not enable buffer 3 when its flag is absent.

Freezing is independent of extension selection. A frozen converged direction
remains in the retained space but no longer produces an ordinary correction.
Periodic generalized reduced solves repair loss of orthonormality; their
frequency is controlled by `GeneralizedSolveInterval`. When such a solve makes
explicit vector images necessary, `ExplicitImageRecalculationCount` records the
recalculation.

## Statistics

`InteriorEigenSolverStatistics` separates operator-call cost from block width:

- `OperatorApplicationCount` counts calls to `ApplyOn`.
- `MultipliedVectorCount` counts the total columns passed to `ApplyOn`.
- `MaximumExpansionSpaceSize` measures the peak number of retained directions.
- `GeneralizedSolveCount` and `ExplicitImageRecalculationCount` expose reduced
  solve maintenance costs.
- Generated extension counts are measured before rank filtering; retained
  counts measure the surviving directions. Additional-Ritz and recycled-vector
  counts are cumulative across collapses.
- Current and maximum frozen-vector counts describe freezing behavior.

For a block-capable operator, one application can multiply several vectors.
Consequently, `MultipliedVectorCount` is normally the more portable proxy for
expensive matrix-vector work, while `OperatorApplicationCount` exposes batching.

## Configuration guidance

The default buffers 0+3 remain the conservative compatibility choice. A Release
comparison on the repository's matrix-free 512-dimensional hub-and-band problem
gave the following single-run results; timings are illustrative rather than a
portable guarantee:

| Extensions | Iterations | Operator applications | Multiplied vectors | Peak expansion | Time (ms) |
|---|---:|---:|---:|---:|---:|
| buffers 0+3 | 151 | 151 | 13,654 | 315 | 9,677 |
| buffers 0+1 | 63 | 125 | 10,992 | 318 | 5,836 |
| buffers 0+2 | 118 | 235 | 19,823 | 316 | 12,277 |
| buffer 0 | 131 | 131 | 11,796 | 224 | 5,768 |
| buffers 0+1+2 | 30 | 59 | 6,762 | 402 | 4,277 |
| all buffers | 19 | 37 | 3,789 | 448 | 2,695 |

Buffer 1 is the safer performance-oriented extension in this comparison. All
extensions delivered the lowest operator work and elapsed time, but retained
87.5% of the full vector-space dimension at peak. Buffer 2 alone was not useful
on this problem. Small problems often converge before richer extensions can
repay their extra work, and the best choice remains operator- and spectrum-dependent.

The hidden Catch2 comparison case reproduces the measurements for real and
complex tridiagonal operators and for two hub-and-band sizes. After building the
test executable in Release mode, run:

```console
MathIterativeVectorInteractionSelfAdjointEigenSolverTest "[.benchmark]"
```

It emits CSV containing timing, convergence, operator-cost, expansion-space,
extension-retention, and residual measurements. Timing is never used as a test
assertion.
