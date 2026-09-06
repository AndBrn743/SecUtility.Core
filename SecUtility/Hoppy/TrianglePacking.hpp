// SPDX-License-Identifier: MIT

#pragma once

#include <Eigen/Core>

#if EIGEN_WORLD_VERSION != 3 || EIGEN_MAJOR_VERSION != 5 || EIGEN_MINOR_VERSION != 0
#error "Hoppy requires Eigen 5.0.0; use a supported Eigen release or update Hoppy's Eigen integration."
#endif

namespace Hoppy
{
	enum class TrianglePacking
	{
		Lower,
		Upper
	};
}  // namespace Hoppy
