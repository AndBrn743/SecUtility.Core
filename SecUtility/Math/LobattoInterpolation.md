# Chebyshev-Lobatto interpolation proposal

Add `LobattoInterpolation.hpp` in `SecUtility::Math`.
This extracts a quadrature-independent interpolation primitive validated locally
in SecGto. It does not add adaptive partitioning or quadrature-family policies.

## Interface

```cpp
template <typename Scalar>
Eigen::VectorX<Scalar> ChebyshevLobattoNodes(Eigen::Index degree);

template <typename Scalar, int DegreeAtCompileTime>
const Eigen::Vector<Scalar, DegreeAtCompileTime + 1>& ChebyshevLobattoNodes();

template <typename Scalar>
Eigen::VectorX<Scalar> ChebyshevLobattoBarycentricWeights(Eigen::Index degree);

template <typename Scalar, int DegreeAtCompileTime>
const Eigen::Vector<Scalar, DegreeAtCompileTime + 1>&
ChebyshevLobattoBarycentricWeights();

template <int DegreeAtCompileTime, typename SampleMatrix>
Eigen::Vector<typename SampleMatrix::Scalar, SampleMatrix::ColsAtCompileTime>
InterpolateChebyshevLobatto(
    typename SampleMatrix::Scalar x,
    const Eigen::MatrixBase<SampleMatrix>& samples);
```

Degree must be positive. The runtime generators return owning vectors; fixed-degree
accessors share immutable, function-local static vectors per scalar/degree
specialization. Nodes and barycentric weights are defined together and selected
internally by interpolation, preventing an arbitrary grid from being paired with
incompatible closed-form weights. No nodes or weights are stored per table object.

Nodes ascend from -1 to 1, with exact endpoints and an exact zero midpoint for even
degrees. The alternating barycentric weights have half-sized endpoints. They are
proportional interpolation weights, not quadrature integration weights. The
Chebyshev name distinguishes this grid from Legendre-Gauss-Lobatto grids.

## Matrix and coordinate contracts

The sample matrix has degree+1 rows and at least one column. Row i stores all
sample values at node i; each column is interpolated independently. Fixed-size
shape violations are diagnosed with static assertions, while dynamic dimensions
are checked with Eigen assertions. Assertions follow the usual Eigen build policy.

Row-major and column-major matrices, blocks, strided maps, and matrix expressions
are accepted. The implementation accesses rows through Eigen and assumes no
contiguous storage. The result is an owning column vector with fixed size when
the sample column count is fixed, and dynamic size otherwise. It remains valid
after an input expression expires or sample storage changes.

The sample scalar determines computation and result precision. The coordinate is
converted to that scalar. Nodes and weights use that same precision; there is no
implicit mixed-precision accumulation policy. Inputs must have suitable finite
floating-point values and x must be finite. The caller maps physical intervals to
[-1,1]. Exact node matches return the stored sample row. Outside [-1,1], the same
formula extrapolates the polynomial, without a numerical accuracy guarantee.

The implementation uses unscaled barycentric terms. Finite inputs and a finite
mathematical result do not guarantee finite intermediate reciprocals, products,
or sums. Very small distances to nodes or large sample magnitudes can therefore
cause overflow and produce infinity or NaN. A possible nearest-node rescaling
is documented beside the accumulation loop in the implementation for future
hardening; it is not currently applied.

## Example using a matrix block

```cpp
#include <SecUtility/Math/ChebyshevLobattoInterpolation.hpp>

using namespace SecUtility::Math;
constexpr int Degree = 3;
const auto& nodes = ChebyshevLobattoNodes<double, Degree>();
Eigen::Matrix<double, 6, 4> storage = Eigen::Matrix<double, 6, 4>::Zero();
auto samples = storage.block<Degree + 1, 2>(1, 1);
for (int i = 0; i <= Degree; ++i)
{
    const double x = nodes[i];
    samples(i, 0) = 1 + x + x*x*x;
    samples(i, 1) = 2 - x*x;
}
const auto values = InterpolateChebyshevLobatto<Degree>(0.25, samples);
// values[0] approximates 1.265625; values[1] approximates 1.9375.
```

## Validation and scope

`MathChebyshevLobattoInterpolationTest` covers exact sample recovery, polynomial
reproduction through the interpolation degree, endpoints and nearby points,
degrees beyond 8/12, Eigen layouts/views/expressions, dynamic sample dimensions,
single-column results, runtime/fixed generator agreement, shared immutable data,
and barycentric weights checked against the independent product definition.
Runtime generator tests also exercise long double with precision-relative bounds.

The corresponding SecGto prototype tests passed according to the user. This port
and its additional tests have not been compiled or run; numerical validation is
pending. Clangd parsed the test translation unit using existing C++17 build
flags with zero diagnostics (the known Catch2 `__COUNTER__` extension warning
was suppressed for this check, and unrelated clangd refactoring checks disabled).
No existing quadrature API or unrelated source is changed.

The interpolation degree remains a compile-time parameter in this proposal.
Runtime-degree interpolator ownership/caching can be designed separately. Generic
piecewise tables, adaptive builders, working precision policy, and quadrature
adapters are later changes.
