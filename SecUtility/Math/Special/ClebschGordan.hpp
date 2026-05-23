// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Math/Special/Factorial.hpp>
#include <SecUtility/Raw/Int.hpp>


namespace SecUtility::Math
{
	namespace Detail::ClebschGordan
	{
		/// <summary>
		/// Calculate and returns the Clebsch-Gordan coefficient with doubled arguments
		/// </summary>
		/// <remarks>
		/// This method will not preform any sanity check! Please consider use
		/// <c>CalculateClebschGordanCoefficientWithDoubledArgs</c> or <c>ClebschGordanCoefficient</c> instead
		/// </remarks>
		/// <param name="j1x2">Represents j<sub>1</sub> × 2</param>
		/// <param name="m1x2">Represents m<sub>1</sub> × 2</param>
		/// <param name="j2x2">Represents j<sub>2</sub> × 2</param>
		/// <param name="m2x2">Represents m<sub>2</sub> × 2</param>
		/// <param name="jx2">Represents J × 2</param>
		/// <param name="mx2">Represents M × 2</param>
		constexpr double unsafe_CalculateClebschGordanCoefficientWithDoubledArgs(const Int64 j1x2,
		                                                                         const Int64 m1x2,
		                                                                         const Int64 j2x2,
		                                                                         const Int64 m2x2,
		                                                                         const Int64 jx2,
		                                                                         const Int64 mx2) noexcept
		{
			if (j1x2 == 0 || j2x2 == 0)
			{
				return jx2 == j1x2 + j2x2 && mx2 == m1x2 + m2x2 ? 1 : 0;
			}

			double factor1 = 1;

			factor1 *= Sqrt(static_cast<double>(jx2 + 1)                //
			                * Math::Factorial((j1x2 + j2x2 - jx2) / 2)  //
			                * Math::Factorial((jx2 + j2x2 - j1x2) / 2)  //
			                * Math::Factorial((jx2 + j1x2 - j2x2) / 2)  //
			                / Math::Factorial((jx2 + j1x2 + j2x2) / 2 + 1));

			factor1 *= Sqrt(Math::Factorial((jx2 + mx2) / 2)      //
			                * Math::Factorial((jx2 - mx2) / 2)    //
			                * Math::Factorial((j1x2 + m1x2) / 2)  //
			                * Math::Factorial((j1x2 - m1x2) / 2)  //
			                * Math::Factorial((j2x2 + m2x2) / 2)  //
			                * Math::Factorial((j2x2 - m2x2) / 2));

			const auto minZ = Max((j2x2 - m1x2 - jx2) / 2, (j1x2 + m2x2 - jx2) / 2, 0);
			const auto maxZ = Min((j1x2 + j2x2 - jx2) / 2, (j1x2 - m1x2) / 2, (j2x2 + m2x2) / 2);

			double factor2 = 0;
			for (auto z = minZ; z <= maxZ; z++)
			{
				double f = z % 2 == 1 ? -1 : 1;

				f /= Math::Factorial(z) * Math::Factorial((j1x2 + j2x2 - jx2) / 2 - z)
				     * Math::Factorial((j1x2 - m1x2) / 2 - z) * Math::Factorial((j2x2 + m2x2) / 2 - z);

				f /= Math::Factorial((jx2 - j2x2 + m1x2) / 2 + z) * Math::Factorial((jx2 - j1x2 - m2x2) / 2 + z);

				factor2 += f;
			}
			return factor1 * factor2;
		}

	}


	/// <summary>
	/// Calculate and returns the Clebsch-Gordan coefficient with doubled arguments
	/// </summary>
	/// <param name="j1x2">Represents j<sub>1</sub> × 2</param>
	/// <param name="m1x2">Represents m<sub>1</sub> × 2</param>
	/// <param name="j2x2">Represents j<sub>2</sub> × 2</param>
	/// <param name="m2x2">Represents m<sub>2</sub> × 2</param>
	/// <param name="jx2">Represents J × 2</param>
	/// <param name="mx2">Represents M × 2</param>
	constexpr double ClebschGordanCoefficientWithDoubledArgs(
	        const Int64 j1x2, const Int64 m1x2, const Int64 j2x2, const Int64 m2x2, const Int64 jx2, const Int64 mx2)
	{

		if (m1x2 + m2x2 != mx2)
		{
			throw InvalidArgumentException("Unphysical m1, m2, and m combination");
		}
		if (jx2 > j1x2 + j2x2 || jx2 < Abs(j1x2 - j2x2))
		{
			throw InvalidArgumentException("Unphysical j1, j2, and j combination");
		}
		if (Abs(m1x2) > j1x2 || Abs(m1x2) % 2 != j1x2 % 2)
		{
			throw InvalidArgumentException("Unphysical j1, m1 combination");
		}
		if (Abs(m2x2) > j2x2 || Abs(m2x2) % 2 != j2x2 % 2)
		{
			throw InvalidArgumentException("Unphysical j2, m2 combination");
		}
		if (Abs(mx2) > jx2 || Abs(mx2) % 2 != jx2 % 2)
		{
			throw InvalidArgumentException("Unphysical j, m combination");
		}

		return Detail::ClebschGordan::unsafe_CalculateClebschGordanCoefficientWithDoubledArgs(
		        j1x2, m1x2, j2x2, m2x2, jx2, mx2);
	}


	/// <summary>
	/// Calculate and returns the Clebsch-Gordan coefficient. The arguments can be either integer or half integer,
	/// they will be rounded to the closest integer or half integer if they are not.
	/// </summary>
	/// <param name="j1">Represents j<sub>1</sub></param>
	/// <param name="m1">Represents m<sub>1</sub></param>
	/// <param name="j2">Represents j<sub>2</sub></param>
	/// <param name="m2">Represents m<sub>2</sub></param>
	/// <param name="j">Represents J</param>
	/// <param name="m">Represents M</param>
	constexpr double ClebschGordanCoefficient(
	        const double j1, const double m1, const double j2, const double m2, const double j, const double m)
	{
		const auto m1x2 = Round<Int64>(m1 * 2);
		const auto m2x2 = Round<Int64>(m2 * 2);
		const auto mx2 = Round<Int64>(m * 2);
		const auto j1x2 = Abs(Round<Int64>(j1 * 2));
		const auto j2x2 = Abs(Round<Int64>(j2 * 2));
		const auto jx2 = Abs(Round<Int64>(j * 2));

		return ClebschGordanCoefficientWithDoubledArgs(j1x2, m1x2, j2x2, m2x2, jx2, mx2);
	}
}
