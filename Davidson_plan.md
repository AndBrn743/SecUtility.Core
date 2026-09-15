# Matrix-Free Davidson Self-Adjoint Eigensolver — Implementation Plan

This work is a redesign and reimplementation, not an API-compatible port of the legacy
`DavidsonDiagonalizer`. Breaking the legacy API is intentional. The new solver will follow the C++20,
matrix-free conventions established by `IterativeVectorInteractionSelfAdjointEigenSolver` and will support both
real symmetric and complex Hermitian operators.

## Gated workflow

Every phase is independently reviewable and includes its own comprehensive tests. Phases are strictly gated:

1. The agent implements only the current phase and writes its tests.
2. The agent may inspect code and use non-executing static-analysis tools such as `clang-tidy`.
3. The agent must not compile any source, build any target, run any test, or run a tool that implicitly compiles or
   executes tests.
4. The agent reports the changed files, static checks performed, checks deliberately not performed, and asks the
   user to compile and run the relevant tests.
5. The phase remains **awaiting user verification** after implementation. It is not complete merely because its code
   and tests have been written.
6. The agent must not begin the next phase until the user explicitly says that the current phase is complete. A test
   result alone is not approval unless the user also closes the phase.

Formatting is outside the scope of individual phases. Agents may ignore `clang-format`; the user will format the
complete implementation after the final phase.

At the start of each phase, the agent should reread this workflow and inspect the current worktree so unrelated user
changes are preserved. If review of a phase changes a later design, update this plan as part of the reviewed phase
rather than silently diverging from it.

## Progress

Only the user may change a phase's status to **Complete**. After implementation and tests have been written, the
agent changes the status to **Awaiting user verification** and stops at the phase gate.

| Phase | Scope | Status |
|---:|---|---|
| 1 | Shared self-adjoint operator protocol | Complete |
| 2 | Public Davidson contracts and input validation | Complete |
| 3 | Initial subspace and vector-image invariants | Complete |
| 4 | Rayleigh-Ritz analysis and inspectable partial results | Complete |
| 5 | Diagonal Davidson correction kernel | Complete |
| 6 | Basic Davidson iteration and terminal semantics | In progress |
| 7 | Thick restart and bounded storage | Not started |
| 8 | Typed correction customization and Olsen correction | Not started |
| 9 | Immutable iteration controller and custom convergence | Not started |
| 10 | Controlled operator mutation for future TRAH use | Not started |
| 11 | Initial-guess policies and dense convenience adapter | Not started |
| 12 | Validation, documentation, and stabilization | Not started |

## Intended public architecture

The target API consists of a typed solver and separate configuration, status, statistics, and inspection types:

```cpp
DavidsonSelfAdjointEigenSolver<Operator> solver;
DavidsonEigenSolverOptions<RealScalar> options(rootCount);

const auto status = solver.Compute(linearOperator, initialBasis, options);
```

The operator supplies dimensions, a real diagonal, and vector or block application. The solver owns no operator and
does not store callbacks that may outlive an operator. It retains aligned subspace vectors and images,
`V` and `W = A V`, and derives the reduced matrix from them. Public inspection of basis vectors, images, the reduced
matrix, and reduced eigenvectors is read-only.

Customization is divided into typed responsibilities:

- root selection determines which reduced Ritz pairs are followed;
- correction generation produces expansion candidates;
- restart selection determines which Ritz directions survive a collapse;
- convergence may add an immutable application-specific criterion;
- an iteration controller observes immutable state and returns typed actions;
- a controller-induced operator change invalidates images explicitly and forces a consistent refresh.

The initial implementation targets the lowest algebraic eigenpairs. Extension points should be designed early, but
additional built-in selectors and a true Jacobi-Davidson inner solve are deferred until the Davidson core is stable.

## Phase 1 — Shared self-adjoint operator protocol

Extract the scalar aliases, self-adjoint operator concepts, and block-versus-column application dispatch currently
embedded in the iVI implementation into a small shared header. Preserve iVI behavior and public API. Define the exact
Davidson operator requirements through this shared protocol: square positive dimension, supported real or complex
scalar, real-valued diagonal, and vector application with optional block application.

Keep statistics policy out of the shared dispatch so each solver can account for applications independently.

Tests must cover compile-time acceptance and rejection of representative operators, real and complex scalar
deduction, block dispatch, column-wise fallback, dimensions, diagonal typing, and unchanged end-to-end iVI behavior
for both block-capable and vector-only operators.

**Gate:** ask the user to build and run the affected matrix-free operator and iVI tests. Phase 2 may begin only after
the user marks Phase 1 complete.

## Phase 2 — Public Davidson contracts and input validation

Introduce the public solver shell and its foundational value types:

- `DavidsonEigenSolverOptions<RealScalar>`;
- `DavidsonEigenSolverStatus`;
- `DavidsonEigenSolverStatistics`;
- result accessors for eigenvalues, eigenvectors, and residual norms;
- read-only final-subspace accessors reserved for downstream algorithms.

Use an explicit `Compute` call; constructors must not execute a solve. Establish reset semantics so every `Compute`
starts cleanly and failed validation cannot expose stale results. Use exceptions for invalid caller input and statuses
for algorithmic terminal outcomes. Initial options should include root count, iteration limit, initial and maximum
subspace dimensions, residual and eigenvalue-change tolerances, preconditioner denominator floor, and linear-
dependence tolerance.

Tests must cover defaults, valid construction, every validation boundary, non-square and inconsistent-diagonal
operators, invalid basis dimensions, non-finite or non-positive tolerances, root/subspace dimension relationships,
status defaults, result alignment, and stale-state clearing across repeated calls. At this phase a valid solve may
terminate through an explicitly documented placeholder path; no Davidson iteration is added yet.

**Gate:** ask the user to build and run the new Davidson contract tests. Phase 3 may begin only after the user marks
Phase 2 complete.

## Phase 3 — Initial subspace and vector-image invariants

Implement initial-basis preparation and the internal aligned subspace representation `(V, W)`. Accept an explicit
initial basis, orthonormalize it with rank detection, reject an empty surviving space, apply the operator in a block
when available, and form the Hermitian reduced matrix `V.adjoint() * W`.

Do not add generated guesses yet. Do not expose mutable internal matrices. Track operator application count,
multiplied-vector count, and maximum subspace size from the first subspace construction.

Tests must cover real and complex bases, already orthonormal bases, scaling, rank deficiency, nearly dependent
columns on both sides of the tolerance, more columns than rows, vector/image alignment, Hermitian projection,
block-versus-column application, exact statistics, and preservation of the operator's input basis.

**Gate:** ask the user to run the Davidson initial-subspace tests. Phase 4 may begin only after the user marks Phase 3
complete.

## Phase 4 — Rayleigh-Ritz analysis and inspectable partial results

Implement the reduced self-adjoint eigensolve, ascending Ritz ordering, lowest-root selection, Ritz-vector and image
formation, residual calculation from retained images, and publication of aligned partial results. Preserve the
reduced coefficient vectors needed by downstream algorithms.

Numerical failure of the reduced eigensolve must produce a precise terminal status without publishing misaligned
data. A valid initial subspace that already spans the requested eigenvectors should be able to converge without an
expansion step.

Tests must compare reduced and lifted results with dense Eigen references for real symmetric and complex Hermitian
matrices; cover exact eigenvectors, rotated full spaces, partial spaces, repeated eigenvalues, ordering, residual
formulas, eigenvector phase/sign independence, numerical-failure handling where constructible, and all final-
subspace accessors.

**Gate:** ask the user to run the Rayleigh-Ritz and exact-subspace tests. Phase 5 may begin only after the user marks
Phase 4 complete.

## Phase 5 — Diagonal Davidson correction kernel

Add the default diagonal-preconditioned residual correction as a standalone internal strategy with a typed,
batch-oriented context. Define and document the sign convention consistently. Regularize denominators whose
magnitude is below the configured floor without discarding their vector components. Orthogonalize candidates
against the retained basis and each other, remove dependent columns, and preserve correspondence to their source
roots for statistics and future policies.

Converged roots must not generate corrections. Zero and dependent corrections must be handled as normal algorithmic
outcomes rather than normalized into NaNs.

Tests must cover exact scalar formulas, both denominator signs, equality and near-equality to diagonal entries,
configured floors, real and complex residuals, multiple roots, converged-root skipping, zero corrections,
cross-correction dependence, reorthogonalization, rank thresholds, finite outputs, and generated-versus-retained
statistics.

**Gate:** ask the user to run the correction-kernel tests. Phase 6 may begin only after the user marks Phase 5
complete.

## Phase 6 — Basic Davidson iteration and terminal semantics

Connect Rayleigh-Ritz analysis and diagonal corrections into an iteration loop without restart. Append accepted
corrections and their operator images while maintaining `(V, W)` alignment. Implement residual convergence,
iteration-limit termination, correction-space exhaustion, numerical-failure propagation, and publication of the
latest Ritz approximations for every non-validation terminal outcome.

Eigenvalue-change checks may supplement residual convergence but must never allow one root's behavior to stand in
for all requested roots. Iteration and operator statistics must have documented, deterministic meanings.

Tests must cover one- and multiple-root convergence; real and complex matrices; diagonal, diagonally dominant,
clustered, and degenerate spectra; exact initial guesses; iteration limits at boundary values; exhausted correction
spaces; repeat `Compute` calls; residual and eigenvalue-change criteria; result ordering and alignment; and exact
statistics on controlled problems.

**Gate:** ask the user to run the basic end-to-end Davidson tests. Phase 7 may begin only after the user marks Phase 6
complete.

## Phase 7 — Thick restart and bounded storage

Implement a thick restart before an append would exceed `MaximumSubspaceDimension`. Retain the requested Ritz
directions and a configurable number of additional Ritz directions, transform both vectors and images with the same
coefficients, and rebuild the reduced matrix from the transformed pair. Validate option combinations that cannot
hold a viable restarted space.

The first version should use a deterministic built-in restart policy while locating it behind a narrow internal
interface suitable for later replacement. Locking or freezing converged roots is not part of this phase unless it is
required to correct a demonstrated convergence defect.

Tests must force single and repeated restarts; verify the dimension bound at every iteration; compare restarted and
unbounded results; verify transformed vector/image alignment; cover extra-retention values, clustered and repeated
roots, rank loss during collapse, exhausted full-dimensional spaces, real and complex operators, and restart and
peak-space statistics.

**Gate:** ask the user to run the restart and bounded-storage tests. Phase 8 may begin only after the user marks Phase
7 complete.

## Phase 8 — Typed correction customization and Olsen correction

Promote correction generation to the public customization model without exposing mutable solver state. A custom
correction strategy receives an immutable context and returns candidate vectors plus source metadata. Keep the
diagonal correction as the default and implement Olsen correction as a built-in strategy using the same denominator
regularization and batch conventions.

Choose template policy, callable parameter, or a small type-erased wrapper based on lifetime safety and usability;
the selected design must not make the solver retain references to temporary customization objects. Record the final
choice and rationale in the public documentation.

Tests must cover default compatibility, stateful custom strategies, temporary/lifetime safety at compile time where
possible, call counts and context contents, malformed dimensions, zero/dependent custom output, exception behavior,
Olsen formulas, real and complex cases, phase invariance, near-singular denominators, and end-to-end convergence for
all built-in corrections.

**Gate:** ask the user to run the correction-customization tests. Phase 9 may begin only after the user marks Phase 8
complete.

## Phase 9 — Immutable iteration controller and custom convergence

Add an immutable `DavidsonIterationInfo` view containing iteration index, basis and images, reduced matrix, selected
Ritz data, residuals, convergence flags, and subspace/restart information. Add a typed controller response supporting
`Continue`, `StopRequested`, and `Restart`. Add an optional custom convergence predicate that can veto built-in
convergence but cannot mutate internal storage.

Define exactly when observation occurs: after a consistent Rayleigh-Ritz analysis and before convergence and
expansion decisions. Define the lifetime of every view as callback-local. A user-requested stop must return a distinct
status and retain the currently published approximations.

Tests must cover callback ordering, every visible field, constness and non-mutability, continue/stop/restart actions,
custom convergence acceptance and veto, first and final iteration behavior, exception propagation, repeated solves,
and absence of callbacks after a terminal decision.

**Gate:** ask the user to run the controller and custom-convergence tests. Phase 10 may begin only after the user marks
Phase 9 complete.

## Phase 10 — Controlled operator mutation for future TRAH use

Extend the controller response with an explicit `OperatorChanged` action. The controller may update state captured by
the operator, but it may not edit the reduced matrix. In response, the solver must reapply the operator to the entire
current basis, rebuild the reduced matrix, rerun Rayleigh-Ritz analysis, and only then evaluate convergence or form
corrections.

Prevent an infinite same-iteration refresh loop with a documented per-iteration refresh limit or an equivalent
deterministic rule. Account separately for refreshes, operator calls, and multiplied vectors. Preserve a coherent
partial result if refresh fails numerically or throws.

Use a small parameterized augmented-Hessian test operator modeled on the needs of TRAH, but do not implement or port
TRAH. Tests must verify that changing its coupling parameter agrees with a fresh dense solve; stale images are never
used; multiple allowed refreshes behave deterministically; refresh limits terminate precisely; block and column-wise
operators agree; exceptions and numerical failures preserve invariants; and all refresh statistics are exact.

**Gate:** ask the user to run the operator-mutation and augmented-Hessian tests. Phase 11 may begin only after the user
marks Phase 10 complete.

## Phase 11 — Initial-guess policies and dense convenience adapter

Add explicit, deterministic initial-guess helpers without coupling guess generation to the solver lifecycle:

- coordinate vectors selected from the smallest diagonal entries;
- identity-prefix vectors;
- seeded random orthonormal vectors;
- augmentation of a caller-provided basis to a requested initial dimension.

Add a lightweight dense self-adjoint operator adapter, or establish that an existing repository abstraction safely
provides the same interface. Helpers must return matrices that callers may inspect or modify before `Compute`.

Tests must cover determinism, seeds, ties in diagonal ordering, dimensions, rank, augmentation without reordering the
caller's independent directions, real and complex scalar types, invalid requests, dense-adapter vector and block
application, diagonal extraction, and end-to-end equivalence between dense-adapted and matrix-free operators.

**Gate:** ask the user to run the guess-helper and dense-adapter tests. Phase 12 may begin only after the user marks
Phase 11 complete.

## Phase 12 — Validation, documentation, and stabilization

Perform the final public-API and naming review. Document algorithm scope, operator requirements, status semantics,
statistics, result ordering, subspace invariants, restart behavior, correction customization, controller timing,
operator refresh, lifetime rules, and a future TRAH integration sketch. Clearly state that arbitrary reduced-matrix
mutation is unsupported.

Add broad validation cases comparing against dense Eigen references across matrix families and sizes, including
random real symmetric and complex Hermitian matrices, diagonal and diagonally dominant matrices, clustered and
repeated eigenvalues, ill-scaled spectra, multiple roots, poor guesses, forced restarts, vector-only operators, and
controller-driven operator changes. Add compile-only API examples where the test framework supports them. Do not add
wall-time assertions or run benchmarks as part of this phase.

Remove transitional scaffolding and placeholder paths. Review headers for accidental dependencies, dangling
references, hidden dense `n x n` operations, unchecked narrowing, and inconsistent exception/status use. Static
analysis remains allowed, but compilation and test execution remain the user's responsibility.

**Final gate:** ask the user to build and run the complete relevant test suite and perform the final review. The work
is not complete until the user explicitly marks Phase 12 complete.

## Deferred work

- reimplementation or migration of TRAH itself;
- a matrix-free Jacobi-Davidson correction equation with configurable inner solver and preconditioner;
- highest-root, closest-to-target, and general interior-root built-in selectors;
- pluggable public restart policies beyond the initial thick-restart policy;
- converged-root locking/freezing unless validation demonstrates it is necessary;
- optimized low-rank image updates after operator mutation;
- generalized eigenproblems;
- automatic dense fallback and performance benchmarking.
