// SPDX-License-Identifier: MIT
// Copyright (c) 2025-2026 Andy Brown

#pragma once

#if __has_include(<cblas.h>) && __has_include(<SecUtility/Math/CBlas.hpp>)
#include <SecUtility/Math/CBlas.hpp>
#endif

#include <Eigen/Dense>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>


namespace SecUtility::Math
{
	template <typename Vector0, typename Vector1, typename Scalar, typename RealScalar>
	std::enable_if_t<std::is_base_of_v<Eigen::MatrixBase<Vector0>, Vector0>
	                         && std::is_base_of_v<Eigen::MatrixBase<Vector1>, Vector1>,
	                 void>
	PerformsRotationOfPointsInThePlane(Vector0&& vector0, Vector1&& vector1, const Scalar sine, const RealScalar cosine)
	{
		eigen_assert(vector0.rows() == 1 || vector0.cols() == 1);
		eigen_assert(vector1.rows() == 1 || vector1.cols() == 1);
		eigen_assert(vector0.size() == vector1.size());

		if constexpr (std::is_convertible_v<Scalar, RealScalar>)
		{
#if __has_include(<cblas.h>) && __has_include(<SecUtility/Math/CBlas.hpp>)
			CBlas::PerformsRotationOfPointsInThePlane<Scalar, RealScalar>(
			        static_cast<CBlas::Integer>(vector0.size()),
			        vector0.derived().data(),
			        static_cast<CBlas::Integer>(vector0.derived().stride()),
			        vector1.derived().data(),
			        static_cast<CBlas::Integer>(vector1.derived().stride()),
			        cosine,
			        sine);
#else
			const auto old = vector0.eval();
			vector0 = cosine * old + sine * vector1;
			vector1 = -sine * old + cosine * vector1;
#endif
		}
		else
		{
			const auto old = vector0.eval();
			vector0 = cosine * old + sine * vector1;
			vector1 = -Conj(sine) * old + cosine * vector1;
		}
	}
}
