# Triangular-compressed Eigen 5 integration (Phase 0)

The Phase 0 spike in `tests/HoppyEigenIntegrationTest.cpp` establishes the extension points used
by the production implementation after owner gate clearance:

- Public Hermitian aliases normalize their structure tag before instantiating the single Detail
  implementation type. This makes real Hermitian/symmetric and real anti-Hermitian/anti-symmetric
  spellings exact type aliases while retaining distinct complex types.
- A dedicated storage kind and shape use Eigen 5's three-parameter `generic_xpr_base`, evaluator,
  `plain_object_eval`, `ref_selector`, storage promotion, unary, binary, transpose, and product
  dispatch points. Specializations remain narrow to Hoppy types.
- Packed maps use partial specializations of Eigen's three-parameter `Map`. Eigen 5's default
  stride is `Stride<0, 0>`; only that exact type receives a specialization. Consequently a custom
  stride has a clean compile-time rejection point without pretending packed storage is dense
  strided storage.
- Mutable and const maps are separate specializations. Map options preserve `Unaligned == 0` and
  `Aligned`; aligned constructors validate the pointer against `EIGEN_MAX_ALIGN_BYTES`, matching
  Eigen's `MapBase::checkSanity` contract.
- Expression operands use `ref_selector`: owning lvalues are referenced and temporary expression
  nodes are retained by value. The spike exercises a nested rvalue unary chain through an Eigen
  evaluator.

This note records extension-point conclusions only. It is not a public API and no production
triangular-compressed class is introduced before the Phase 0 owner gate is cleared.
