# Hoppy v1 traceability

This checklist maps the normative decisions in `BlockDiagonalMatrix-Spec.md` to regression
tests or explicit review items. Test names refer to files under `tests/`.

| Decisions | Evidence |
|---|---|
| D1–D5 | `HoppyFactoryTest`, `HoppyContractTest`, public-header self-containment target |
| D6, D12, D31–D32 | Throwing assertion targets across `Hoppy*Test`; `HoppyStorageTest`; `HoppyContractTest` |
| D7–D8 | `HoppyContractTest` operation matrix and dynamic-dimension assertions |
| D9 | Explicit non-goal; no binary-I/O API |
| D10, D15, D25 | `HoppyMatvecTest`, `HoppyDenseProductTest`, `HoppyPropertyTest` |
| D11, D21, D28 | `HoppyHeaderSelfContainment`, `HoppyUnsupportedEigenVersion` |
| D13 | `HoppyStorageTest`, `HoppyFactoryTest` |
| D14 | `HoppyFactoryTest`; only `DenseBlockPolicy` is accepted |
| D16 | `HoppyCongruenceTest`, `HoppyPropertyTest` |
| D17 | `HoppyValueCategoryTest` plus lifetime cases in expression-specific tests |
| D18 | `HoppyStorageTest`, `HoppyCwiseTest`, `HoppyBlockProductTest`, `HoppyDiagonalViewTest` |
| D19, D33 | Type assertions and negative detection in cwise/product/solve/vector/contract tests |
| D20, D23 | `HoppyFactoryTest`, `HoppyStorageTest`, `HoppyComplexityTest` |
| D22, D26–D27 | `HoppyFactoryTest`, throwing-scalar cases in `HoppyContractTest` |
| D24, D30 | `HoppyUnaryReductionTest`, `HoppyVectorAlgebraTest` |
| D29 | `HoppyUnaryReductionTest`, `HoppyEvaluationTest`, `HoppyCwiseTest` |

## Specification §16 checklist

| §16 item | Evidence |
|---:|---|
| 1–3 | `HoppyStorageTest`, `HoppyFactoryTest` |
| 4–6 | Cwise, product, matvec, dense-product, dense-sum, and seeded property tests |
| 7 | `HoppyComplexityTest`; structural operation count plus Eigen no-malloc guard |
| 8 | `HoppyDiagonalViewTest`, `HoppyValueCategoryTest` |
| 9–10 | `HoppyUnaryReductionTest`, `HoppyVectorAlgebraTest` |
| 11–12 | `HoppySolveTest`, `HoppyCongruenceTest` |
| 13–14 | Assignment tests in storage/cwise/product/view tests; throwing scalar contract tests |
| 15 | `HoppyValueCategoryTest` and expression-specific prvalue-chain tests; sanitizer gate |
| 16–19 | Assertion tests, policy tests, header targets, and `HoppyContractTest` |
| 20 | `HoppyPropertyTest` uses a fixed seed and empty/unit/mixed blockings |
| 21 | This checklist and the final owner review |

## Manual release review

- Eigen 5.0.0 is guarded in every public include path; Eigen assertion macros must be
  translation-unit consistent.
- Plain-object maps and views follow the invalidation rules in specification §§4 and 8.
- Lazy expressions follow Eigen lifetime rules; owning-object rvalues are rejected where a
  node would retain a reference, while intermediate expression nodes nest by value.
- Reblocking commits by swap; allocation and user-scalar exceptions propagate as documented.
- Dense sum/difference uses the approved lazy `ReturnByValue` design and does not densify the
  block-diagonal operand.
- Unsupported features remain those listed in specification §18.
