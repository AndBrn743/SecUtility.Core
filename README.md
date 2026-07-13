# SecUtility.Core

[![CI](https://github.com/AndBrn743/SecUtility.Core/actions/workflows/linux.yml/badge.svg?branch=master)](https://github.com/AndBrn743/SecUtility.Core/actions)
[![CI](https://github.com/AndBrn743/SecUtility.Core/actions/workflows/windows.yml/badge.svg?branch=master)](https://github.com/AndBrn743/SecUtility.Core/actions)
[![CI](https://github.com/AndBrn743/SecUtility.Core/actions/workflows/macos.yml/badge.svg?branch=master)](https://github.com/AndBrn743/SecUtility.Core/actions)
[![codecov](https://codecov.io/gh/AndBrn743/SecUtility.Core/branch/master/graph/badge.svg)](https://codecov.io/gh/AndBrn743/SecUtility.Core)

A header-only C++ utility library. The `Core` slice collects general-purpose
numerical, metaprogramming, threading, and IO primitives that are reused across
larger projects, packaged as a CMake `INTERFACE` target for easy consumption.

- **License:** MIT
- **Language:** C++17 minimum. A small number of features require C++20 or later.
- **Form factor:** Header-only; shipped as a CMake `INTERFACE` library
- **Namespace:** `SecUtility::`
- **CMake target:** `SecUtility::Core`

> **Stability notice:** SecUtility.Core makes no API or ABI stability
> guarantees. All `Sec*` projects by the same author live at head -- there
> are no versioned releases, and the `master` branch may break consumers at
> any time. When consuming via FetchContent, pin to a specific commit and be
> prepared to update your own code when you bump.

## Modules

Top-level subdirectories under `SecUtility/`:

| Module       | Contents                                                                                                                                 |
|--------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `Algorithm`  | Numerical algorithms (e.g. spectator reduction).                                                                                         |
| `Collection` | Containers and iteration helpers (`Array`, `KeyedArray`, `MultidimensionalArray`, `RoundRobinOrdering`, `TypeTuple`, ...)                |
| `Diagnostic` | Error handling, timing, and type introspection (`Exception`, `Stopwatch`, `TypeName`, `Unreachable`)                                     |
| `IO`         | Streaming and serialisation abstractions (to be export)                                                                                  |
| `Macro`      | Small helper macros (`ConstevalIf`, `ForceInline`, `Stringify`)                                                                          |
| `Math`       | Numerical primitives -- quadrature, special functions, continued fractions, structured matrices, compensated summation, etc.             |
| `Meta`       | Compile-time utilities (`IntegerSequence`, `NttpArray`, `ParameterPackUtility`, `OverloadSet`, `StaticForeach`, `TypeTrait`, `Identity`) |
| `Misc`       | Miscellaneous utilities (`Bitflag`, `CachedFunction`, `Checksum`, `Enum`, `NullMutex`, `Prefetch`, `Random`)                             |
| `Raw`        | Low-level wrappers around built-in primitives (`Array`, `Float`, `Int`, `Ptr`, `LRef`, `RRef`)                                           |
| `Text`       | String utilities (`CaseConversion`, `Comparison`, `Conversion`, `Split`, `Symbol`)                                                       |
| `Threading`  | `ThreadPool`                                                                                                                             |

## Dependencies

SecUtility.Core links the following external libraries as `INTERFACE`
dependencies. When the project is built standalone, each is fetched
automatically via CMake's `FetchContent` and (where needed) patched from
`patches/`:

| Dependency                                          | Version | Notes                                                                            |
|-----------------------------------------------------|---------|----------------------------------------------------------------------------------|
| [Eigen](https://gitlab.com/libeigen/eigen)          | 5.0.0   | Linear algebra. Fetched only if `Eigen3::Eigen` is not already defined upstream. |
| [range-v3](https://github.com/ericniebler/range-v3) | 0.12.0  | Ranges library. Auto-patched via `patches/range-v3/`.                            |
| [gcem](https://github.com/kthohr/gcem)              | 1.18.0  | `constexpr` math. Auto-patched via `patches/gcem/`.                              |
| [Catch2](https://github.com/catchorg/Catch2)        | 3.15.1  | Tests only. See [Building the tests](#building-the-tests).                       |

If your top-level project already defines `Eigen3::Eigen`, `range-v3::range-v3`,
or `gcem::gcem` before pulling in SecUtility.Core, those existing targets are
reused as-is and no patch is applied, you take responsibility for the version.

## Consuming the library

### Option A: FetchContent (recommended)

```cmake
include(FetchContent)
FetchContent_Declare(
    secutility_core
    GIT_REPOSITORY https://github.com/AndBrn743/SecUtility.Core.git
    GIT_TAG master  # or pin to a specific commit for reproducible builds
)
FetchContent_MakeAvailable(secutility_core)

target_link_libraries(my_target PRIVATE SecUtility::Core)
```

This transitively fetches Eigen, range-v3, and gcem unless you have already
defined those targets upstream. The test suite is not built by default when
SecUtility.Core is consumed as a subproject, see below to opt in.

### Option B: Install

```bash
cmake -S . -B build
cmake --build build      # nothing to compile (INTERFACE library)
sudo cmake --install build
```

This installs the headers under `${CMAKE_INSTALL_INCLUDEDIR}/SecUtility/` and
an exported CMake targets file under `share/SecUtility.Core/cmake/`.

> Note: a complete `find_package(SecUtility)` config-file story is not yet
> wired up. The exported targets file (`SecUtilityCoreTargets.cmake`) is
> installed, but consumers currently need to `include()` it directly (or add
> the installation directory to `CMAKE_PREFIX_PATH` and `find_package` the
> individual upstream dependencies Eigen, range-v3, gcem themselves). For the
> smoothest experience, prefer FetchContent.

## Building the tests

Tests are gated by the `SECUTILITY_CORE_ENABLE_TEST` CMake option. It defaults
to **ON** when SecUtility.Core is the top-level project and **OFF** when
consumed via `add_subdirectory` or `FetchContent`.

```bash
cmake -S . -B build -DSECUTILITY_CORE_ENABLE_TEST=ON
cmake --build build
ctest --test-dir build/tests --output-on-failure
```

The test suite is written with Catch2 and is fetched via `FetchContent` by
default. To use a system-installed Catch2 instead:

```bash
cmake -S . -B build -DSECUTILITY_CORE_ENABLE_TEST=ON -DUSE_SYSTEM_CATCH2=ON
```

## Tested platforms

CI builds and runs the test suite on the following configurations (all with
`CMAKE_CXX_STANDARD=17`, `CMAKE_CXX_EXTENSIONS=OFF`):

| OS      | Compiler       |
|---------|----------------|
| Linux   | GCC, Clang     |
| macOS   | Xcode Clang    |
| Windows | MSVC, Clang-CL |

## Contributing

Issues and pull requests are welcome. The project is maintained by a single
developer on a best-effort basis, so response times may vary.

## License

MIT, see [LICENSE](LICENSE).
