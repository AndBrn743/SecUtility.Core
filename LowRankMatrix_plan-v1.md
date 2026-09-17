# Eigen-Compatible Low-Rank Matrices — Implementation Plan

This work is a redesign and reimplementation of the prototype
`SecUtility/Hoppy/LowRankMatrix.hpp`, not an API-compatible port. Breaking the prototype API is intentional. The new
types will be C++17-compatible Eigen extensions in namespace `Hoppy`, will support general rectangular, symmetric,
and self-adjoint matrices, and will represent only accumulated low-rank updates. Redesign of
`QuasiNewtonSolver` is deferred to a separate plan after this foundation is complete.

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
| 1 | General low-rank value type, storage, and invariants | Complete |
| 2 | Symmetric and self-adjoint low-rank value types and structural guarantees | Complete |
| 3 | Eigen products, dense inspection, and numerical queries | Complete |
| 4.1 | Transpose, conjugate, and adjoint expressions | Complete |
| 4.2 | Scalar multiplication and division expressions | Complete |
| 5 | Low-rank arithmetic and controlled mutation | Complete |
| 6 | Lazy dense/low-rank sums and Eigen iterative-solver integration | Complete |
| 7 | Optional C++20 SecUtility.Core adapters | Not started |
| 8 | Validation, documentation, licensing, and stabilization | Not started |

## Intended public architecture

The general matrix stores an exact ordered expansion

\[
A = U\,\operatorname{diag}(c)\,V^*,
\]

where `U` is `rows x termCount`, `V` is `cols x termCount`, and `c` has scalar type `Scalar`. The self-adjoint matrix
stores

\[
H = U\,\operatorname{diag}(r)\,U^*,
\]

where `r` has type `Eigen::NumTraits<Scalar>::Real`. A symmetric matrix instead stores
`S = U * diag(c) * U.transpose()` with scalar coefficients. For real scalars, the self-adjoint and symmetric public
aliases name the same type.

The target public family is:

```cpp
Hoppy::LowRankMatrix<Scalar, RowsAtCompileTime, ColsAtCompileTime>
Hoppy::LowRankMatrixX<Scalar>

Hoppy::LowRankSelfAdjointMatrix<Scalar, DimensionAtCompileTime>
Hoppy::LowRankSelfAdjointMatrixX<Scalar>
Hoppy::LowRankSymmetricMatrix<Scalar, DimensionAtCompileTime>
Hoppy::LowRankSymmetricMatrixX<Scalar>
```

The exact alias set may be narrowed if an alias cannot express its invariant clearly in C++17. Public names and
member functions follow Eigen conventions: `rows`, `cols`, `termCount`, `addTerm`, `addTerms`, `reserve`, `toDense`,
`diagonal`, `norm`, and `squaredNorm`.

Both fixed and dynamic dimensions are supported. Dynamic rank-zero values are constructed with explicit dimensions;
default construction produces a `0 x 0` value only where the compile-time dimensions are dynamic. Fixed and runtime
dimensions must agree. General matrices may be rectangular; symmetric and self-adjoint matrices must be square.

Each exactly zero coefficient is ignored on insertion, including in batch insertion. No tolerance, compression,
dependency detection, cancellation, reordering, or term removal is performed in v1. Every nonzero accepted term is
retained in insertion order until the value is cleared or destroyed.

Bulk storage views are read-only. Mutation is limited to invariant-preserving operations such as `addTerm`,
`addTerms`, `reserve`, `clear`, same-scalar in-place low-rank addition/subtraction, and valid scalar scaling. Any view
returned by the API must document whether appending terms invalidates it.

Invalid dimensions, indices, capacities, and expression shapes are programming errors guarded by `eigen_assert`, in
the same style as Eigen. If `EIGEN_NO_DEBUG` disables those checks, violating the corresponding preconditions is
undefined behavior. Exceptions are reserved for resource or representability failures such as allocation failure or
buffer-size overflow.

The types derive from `Eigen::EigenBase` and live in namespace `Hoppy`. They do not derive from or otherwise depend on
`SecUtility::Math::LinearOperatorBase`. Eigen interoperability is deliberately bounded to the operations specified by
this plan; the types must not claim sparse coefficient access, compressed storage, or iterator capabilities that they
do not implement.

## Phase 1 — General low-rank value type, storage, and invariants

Implement the C++17 general rectangular value type and its Eigen traits. Store coefficients, left vectors, and right
vectors in contiguous owned buffers, with fixed or dynamic row and column dimensions. Provide default and explicit
dimension constructors, single-term and batch construction, copy/move operations, read-only coefficient/vector/term
access, `termCount`, `rows`, `cols`, `reserve`, `clear`, `addTerm`, and `addTerms`.

Introduce `LowRankMatrixBase<Derived>` as the shared CRTP base for owning matrices and later lazy low-rank
expressions. It derives from `Eigen::EigenBase<Derived>` and owns the common read-only factor/term interface. Storage
mutation remains on owning derived types so general and structured invariants cannot be bypassed through the base.

Define each term as `coefficient * leftVector * rightVector.adjoint()`. Validate all dimensions before mutating
storage. Make insertion safe when input expressions alias the destination's current read-only views. Ignore exactly
zero coefficients without inspecting their vectors. Batch insertion preserves the relative order of nonzero terms.
Provide a documented strong exception guarantee where practical; at minimum, asserted preconditions are checked
before mutation and failed input evaluation must not leave mismatched buffers or broken dimensions.

Do not add matrix products, dense conversion, transformations, arithmetic expressions, single-factor storage, or Core
adapters in this phase.

Tests must cover fixed, partially dynamic, and fully dynamic dimensions; square and rectangular shapes; rank-zero
construction; single and batch insertion; exact-zero filtering; term order; complex coefficients and vectors;
copy/move behavior; reserve and clear; read-only view types; self-aliasing insertion; asserted invalid dimensions and
shapes; and stable invariants after assertion failures. Add compile-time checks for advertised Eigen traits and for
the absence of mutable bulk access. Test assertion failures in a dedicated target that overrides `eigen_assert`; keep
the ordinary valid-behavior target under Eigen's normal assertion configuration.

## Phase 2 — Symmetric and self-adjoint low-rank value types and structural guarantees

Implement symmetric and self-adjoint value types through the internal policy-driven
`Detail::SingleFactorLowRankMatrix`, which owns one vector buffer. Structural policies own coefficient typing and the
right-factor transformation so the owner does not branch on structure. Symmetric matrices use `Scalar` coefficients
and plain transpose, including for complex scalars. Self-adjoint matrices use `RealScalar` coefficients and conjugate
transpose. For real scalars, the two public aliases must name exactly the same symmetric-policy specialization.

Provide the same construction, capacity, clearing, insertion, and read-only inspection facilities applicable to a
square structured value. Provide an explicit conversion or materialization path to general low-rank form without
dense expansion. Do not accept complex coefficients for a genuinely complex self-adjoint matrix through implicit
conversion. Preserve dimensions when clearing all terms, including self-subtraction in a later phase.

Tests must cover `float`, `double`, `std::complex<float>`, and `std::complex<double>` vector scalars; coefficient
typing; compile-time and runtime dimensions; exact symmetry and self-adjointness of represented expansions; real
alias identity; complex-symmetric matrices; rank-zero values; zero filtering; alias-safe insertion; conversion to the
general form; and compile-time rejection of invalid self-adjoint coefficient combinations.

## Phase 3 — Eigen products, dense inspection, and numerical queries

Add Eigen-compatible multiplication of both value families by dense vectors and dense matrices. Products must honor
Eigen's `scaleAndAddTo` contract, including non-unit `alpha`, aliasing destinations, mixed compatible right-hand scalar
types, fixed/dynamic result dimensions, zero terms, and zero-column blocks. Evaluate products through the low-rank
factors without dense matrix materialization and without a per-term type-erased callback chain.

Add `toDense`, `row`, `col`, and `diagonal`. Add Frobenius `squaredNorm` and `norm`, returning `RealScalar` and handling
complex conjugation correctly. These inspection operations must preserve rectangular dimensions and return correctly
oriented Eigen objects. Avoid pretending that the type supplies general coefficient access or sparse iteration.

Tests must compare every operation against independently constructed dense references for real and complex scalars,
fixed and dynamic dimensions, rectangular and square matrices, rank zero, rank one, repeated/cancelling terms, dense
vector and block right-hand sides, product scaling, destination aliasing, and row/column/diagonal extraction. Norm
tests must use two-sided approximate comparisons and include complex general, symmetric, and self-adjoint cases.

## Phase 4.1 — Transpose, conjugate, and adjoint expressions

Implement Eigen-style `transpose()`, `conjugate()`, and `adjoint()` without dense materialization. General transforms
must preserve the canonical `U * diag(c) * V.adjoint()` representation and swap compile-time/runtime dimensions where
required. Structural policies determine transpose, conjugate, and adjoint behavior without policy-type branching in
the single-factor owner. Self-adjoint `adjoint()` remains structurally self-adjoint; self-adjoint transpose and
conjugate are equivalent structured transformations. Symmetric `transpose()` remains structurally symmetric.

Prefer lazy Eigen-style views/expressions for transformations where their ownership and nesting rules are
safe. Expressions formed from temporaries must not dangle. Assignment/materialization into the owning low-rank types
must be available where the represented form remains low rank.

Tests must cover mathematical equivalence to dense transpose/conjugate/adjoint; rectangular dimension swaps; complex
coefficient conjugation; involutions and identities such as `adjoint().adjoint()`; symmetric and self-adjoint
structural preservation; lvalue and temporary lifetimes; and expression composition with vector/block products.

## Phase 4.2 — Scalar multiplication and division expressions

Implement `lowRank * scalar`, `scalar * lowRank`, and `lowRank / scalar`. General matrices accept compatible scalar
scaling with a well-defined promoted result type. Symmetric matrices preserve their structure under compatible scalar
scaling. A complex self-adjoint matrix scaled or divided by a real scalar remains self-adjoint; scaling or division by
a complex scalar remains low rank but produces a generic low-rank expression. The structural policy selects this
result at compile time, so no explicit conversion is required. Division by zero follows the underlying Eigen/scalar
convention and must be documented rather than silently changing structure.

Prefer lazy Eigen-style expressions where their ownership and nesting rules are safe. Expressions formed from
temporaries must not dangle. Assignment/materialization into the owning low-rank types must be available where the
represented form remains low rank.

Tests must cover scaling equivalence to dense references, scalar promotion, lvalue and temporary lifetimes,
expression composition with vector/block products, and self-adjoint fallback to a generic low-rank expression under
complex scaling.

## Phase 5 — Low-rank arithmetic and controlled mutation

Implement same-scalar in-place `+=` and `-=` for identical owning types. Implement non-mutating low-rank addition and
subtraction for compatible shapes: equal single-factor structures preserve their structure, while general/structured
or differently structured operands produce a general result. Preserve left-to-right term order and implement
subtraction by coefficient negation. Handle self-addition without duplicating vector storage unnecessarily where a
coefficient update suffices; handle self-subtraction by clearing terms while preserving dimensions.

Mixed scalar in-place operations are not supported. Non-mutating mixed-scalar arithmetic may be omitted unless it is
required by Eigen expression interoperability discovered during implementation; any addition must be reviewed before
expanding this phase. Operations must validate shape compatibility before mutation and must remain safe when operands
or exposed views alias.

Tests must cover every supported type combination, rectangular compatibility, fixed/dynamic dimensions, rank-zero
operands, zero coefficients created by arithmetic, term ordering, self-addition, self-subtraction, aliasing, rejected
shape mismatches, structural preservation, conversion of mixed structural arithmetic to general form, and equivalence
to dense references.

## Phase 6 — Lazy dense/low-rank sums and Eigen iterative-solver integration

Implement lazy Eigen-compatible expressions for `lowRank + dense`, `lowRank - dense`, `dense + lowRank`, and
`dense - lowRank`. Follow Eigen nesting rules so expressions safely own or reference operands as appropriate and do
not dangle when either operand is temporary. Support dense materialization, vector products, block products, and
scaled product accumulation without first materializing the low-rank operand.

Expose an efficient combined `diagonal()` for dense-plus-low-rank expressions. Make square compatible expressions
usable by applicable Eigen iterative solvers without falsely advertising the operands as ordinary sparse matrices.
Because Eigen's stock `DiagonalPreconditioner` requires sparse `InnerIterator` access rather than a `diagonal()` API,
provide an Eigen-compatible Hoppy preconditioner derived from it that initializes through the combined diagonal. The
initial dense matrix and low-rank correction remain separate values; the expression is their non-owning or safely
nested algebraic composition.

Tests must cover all four operand orders/signs; fixed/dynamic and complex scalar combinations; rectangular dense
materialization and products; expression nesting and temporary lifetimes; destination aliasing; combined diagonals;
and at least one representative Eigen iterative solve using a dense initial matrix plus a low-rank correction. Compare
solver results and residuals with the equivalent dense matrix, including a complex self-adjoint case where supported by
Eigen.

## Phase 7 — Optional C++20 SecUtility.Core adapters

Add adapters available only in C++20 mode without changing the C++17 Hoppy implementation or its public contracts.
A general adapter supplies Core-style dimensions and `ApplyOn`. A self-adjoint adapter additionally supplies a
real-valued `Diagonal` and satisfies the existing `SecUtility::Math::SelfAdjointLinearOperator` protocol. Adapt the
dense-plus-low-rank expression where useful to the later quasi-Newton solver, but do not make Hoppy types derive from
`LinearOperatorBase` and do not introduce a C++20 dependency into their headers.

Keep adapters non-owning only when lifetime requirements are explicit and safe; otherwise use Eigen-style nested
storage or owned values. Do not add callbacks or `std::function` type erasure merely to bridge naming conventions.

Tests must be compiled explicitly as C++20 and cover concept acceptance/rejection, real and complex scalar aliases,
vector and block application, real self-adjoint diagonals, dense-plus-update adaptation, adapter lifetime behavior, and
unchanged direct use of the underlying C++17 Hoppy API. The main low-rank test target remains C++17.

## Phase 8 — Validation, documentation, licensing, and stabilization

Review the complete implementation for API consistency, Eigen extension correctness, compile-time dimension
propagation, expression lifetimes, aliasing, exception safety, numerical correctness, header self-sufficiency, warning
cleanliness, and accidental C++20 usage in the C++17 surface. Add focused regression tests for every defect found
during earlier user verification.

Document the supported Eigen compatibility boundary, canonical factorization, conjugate-transpose convention,
self-adjoint coefficient restriction, exact-zero policy, insertion order, capacity and invalidation rules, complexity,
dynamic rank-zero construction, scalar promotion, and operations deliberately deferred to v2. Explicitly defer
compression, tolerance-based pruning, dependency detection, term removal, bounded history, and recompression.

Resolve licensing before finalizing public headers. Verify Eigen's applicable license and the provenance of any code
adapted from Eigen or the prototype; add accurate per-file SPDX identifiers and required notices. Namespace `Hoppy`
is an architectural/API boundary, not a substitute for file-level license compliance.

The final verification matrix must include GCC, Clang, and MSVC where available; C++17 for the owning types and Eigen
expressions; C++20 for the optional Core adapters; real and complex scalars; fixed and dynamic dimensions; debug and
release assertion modes; and sanitizer builds suitable for detecting buffer, aliasing, and lifetime errors. Update
test registration and installation coverage so all public headers are exercised as installed, standalone includes.
