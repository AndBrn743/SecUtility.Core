# Davidson lowest-eigenpair solver

`DavidsonSelfAdjointEigenSolver` computes one or more lowest algebraic eigenpairs of a real symmetric or complex
Hermitian operator. It is intended for problems where applying the operator is substantially cheaper than storing or
diagonalizing its full matrix. The implementation requires C++20.

The operator must satisfy `SelfAdjointLinearOperator`: it supplies square dimensions, a finite real diagonal, and
`ApplyOn` for either individual vectors or blocks. Block application is preferred when both forms exist. The solver
does not own the operator and does not retain it or any callback after `Compute` returns. `DenseSelfAdjointLinearOperator`
provides an owning convenience adapter when a dense matrix is already available.

```cpp
DenseSelfAdjointLinearOperator denseOperator(matrix);
DavidsonEigenSolverOptions<double> options(2);
auto initialBasis = CoordinateDavidsonInitialBasis<double>(denseOperator.Diagonal(), 2);
DavidsonSelfAdjointEigenSolver<decltype(denseOperator)> solver;
const DavidsonEigenSolverStatus status = solver.Compute(denseOperator, initialBasis, options);
```

## Results and terminal states

Eigenvalues are returned in ascending algebraic order. Eigenvector column and residual-norm entry `i` correspond to
eigenvalue `i`. Every algorithmic terminal status retains the latest coherent approximation when one exists:

- `Converged`: every requested residual satisfies `ResidualNormTolerance` and the optional convergence predicate
  accepts the result.
- `IterationLimitReached`: the configured iteration count was consumed.
- `ExpansionSpaceExhausted`: correction generation produced no independent usable direction.
- `NumericalFailure`: the reduced self-adjoint solve could not produce a valid analysis.
- `StoppedByController`: the controller requested an orderly stop after observing the current analysis.
- `StructuredOperatorUpdateLimitReached`: the final allowed same-iteration update was applied and reanalyzed.
- `NotComputed`: no solve has completed or the solver was reset by a new call.

Invalid input clears all previous observable state before throwing. Exceptions from user callbacks propagate; the
status remains `NotComputed`. A callback exception may leave the most recently published coherent analysis available
for diagnosis. A rejected structured update clears Ritz results because the callback may already have changed its
captured operator, making the preceding analysis potentially stale.

Residual convergence is authoritative. `EigenvalueChanges` is exposed to controllers and custom convergence
predicates as diagnostic/application-specific information; `EigenvalueChangeTolerance` does not override residual
convergence. A custom convergence predicate is consulted only after all requested roots satisfy the built-in residual
criterion, so it may veto convergence but cannot accept an unconverged result.

## Subspace and restart invariants

The solver retains aligned matrices `V` and `W = A V`. `V` has orthonormal columns, `W` contains their operator
images, and the reduced matrix is `V.adjoint() * W`. Public access to these matrices is read-only. Ritz vectors,
images, residuals, and reduced eigenvectors are derived from the same Rayleigh–Ritz analysis.

When an expansion would exceed `MaximumSubspaceDimension`, thick restart retains every requested Ritz direction
before up to `AdditionalRestartRitzVectorCount` optional directions. A controller may also request restart. Restart
transforms vectors and cached images with identical coefficients, preserving `W = A V` without new operator work.

## Custom correction and iteration control

The default `DiagonalDavidsonCorrection` uses the supplied operator diagonal. `OlsenDavidsonCorrection` is also
provided. A custom correction callable receives an immutable `DavidsonCorrectionContext` and returns
`DavidsonCorrectionCandidates`; returned source indices, dimensions, and finite values are validated before the
candidates are orthogonalized.

The iteration controller runs after a coherent Rayleigh–Ritz analysis and before convergence, restart, or expansion
decisions. `DavidsonIterationInfo` references are immutable and valid only for that callback invocation. Controllers
may return `Continue`, `StopRequested`, or `Restart` directly. A typed `DavidsonIterationDecision` additionally
supports an exact low-rank operator update.

For a self-adjoint change

```text
delta(A) = U C U.adjoint()
```

return `ApplyLowRankOperatorUpdate` with `Factors = U` and self-adjoint `Core = C`. The solver updates the cached
basis images, projection, and diagonal algebraically, then reruns Rayleigh–Ritz analysis in the same iteration. This
path performs no operator application. Invalid dimensions, non-finite data, and non-self-adjoint cores are rejected.
Arbitrary direct mutation or replacement of the reduced matrix is unsupported because it would break consistency
with cached images and residuals.

For a future TRAH integration, changing augmented-Hessian gradient scaling by `delta` can be represented as a rank-two
update with factors `[e0, g]` and core `[[0, delta], [delta, 0]]`. The TRAH controller can perform its reduced-space
scaling search locally, update its captured scaling once, and return this exact delta without repeating expensive
Hessian-vector products. Porting TRAH itself remains deferred.

## Initial guesses and statistics

`DavidsonInitialGuess.hpp` provides inspectable matrices for identity-prefix, smallest-diagonal coordinate, seeded
random orthonormal, and order-preserving caller-basis augmentation policies. Guess construction is independent of the
solver lifecycle.

`OperatorApplicationCount` counts calls to `ApplyOn`; `MultipliedVectorCount` counts all columns passed to it.
`MaximumSubspaceDimension`, `RestartCount`, generated/retained correction counts, and
`StructuredOperatorUpdateCount` expose the remaining algorithmic work. Structured updates deliberately do not alter
either operator-application statistic.
