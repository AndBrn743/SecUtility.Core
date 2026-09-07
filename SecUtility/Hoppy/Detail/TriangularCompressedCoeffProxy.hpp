// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>

#include <Eigen/Core>

namespace Hoppy::Detail
{
	template <typename Matrix>
	class TriangularCompressedCoeffProxy
	{
	public:
		using Scalar = typename Matrix::Scalar;

		TriangularCompressedCoeffProxy(Matrix& matrix, const Eigen::Index row, const Eigen::Index column)
			: m_MatrixPtr(&matrix), m_Row(row), m_Column(column)
		{}

		/* IMPLICIT */ operator Scalar() const { return m_MatrixPtr->coeff(m_Row, m_Column); }

		TriangularCompressedCoeffProxy& operator=(const Scalar& value)
		{
			m_MatrixPtr->writeLogical(m_Row, m_Column, value);
			return *this;
		}

		TriangularCompressedCoeffProxy& operator=(const TriangularCompressedCoeffProxy& other)
		{
			const Scalar value = static_cast<Scalar>(other);
			return *this = value;
		}

		template <typename OtherMatrix>
		TriangularCompressedCoeffProxy& operator=(
		        const TriangularCompressedCoeffProxy<OtherMatrix>& other)
		{
			const Scalar value = static_cast<Scalar>(other);
			return *this = value;
		}

		template <typename Value>
		TriangularCompressedCoeffProxy& operator+=(const Value& value)
		{
			const Scalar result = static_cast<Scalar>(*this) + value;
			return *this = result;
		}

		template <typename Value>
		TriangularCompressedCoeffProxy& operator-=(const Value& value)
		{
			const Scalar result = static_cast<Scalar>(*this) - value;
			return *this = result;
		}

		template <typename Value>
		TriangularCompressedCoeffProxy& operator*=(const Value& value)
		{
			const Scalar result = static_cast<Scalar>(*this) * value;
			return *this = result;
		}

		template <typename Value>
		TriangularCompressedCoeffProxy& operator/=(const Value& value)
		{
			const Scalar result = static_cast<Scalar>(*this) / value;
			return *this = result;
		}

	private:
		Matrix* m_MatrixPtr;
		Eigen::Index m_Row;
		Eigen::Index m_Column;
	};
}  // namespace Hoppy::Detail
