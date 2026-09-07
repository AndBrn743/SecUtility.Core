// SPDX-License-Identifier: MIT

#include <SecUtility/Hoppy/TriangularCompressedMatrixExpr.hpp>

#include <type_traits>

namespace
{
	struct ExpressionHeaderProbe;
	using ProbeBase = Hoppy::TriangularCompressedMatrixExpr<ExpressionHeaderProbe>;
	static_assert(std::is_class_v<ProbeBase>);
}
