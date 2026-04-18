// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <type_traits>


namespace SecUtility::Math
{
	template <typename Derived>
	class LinearOperatorBase;
}


template <typename Derived>
struct Eigen::internal::traits<SecUtility::Math::LinearOperatorBase<Derived>> : traits<Derived>
{
	/* NO CODE */
};


template <typename Derived, typename Rhs>
struct Eigen::internal::generic_product_impl<SecUtility::Math::LinearOperatorBase<Derived>,
                                             Rhs,
                                             Eigen::SparseShape,
                                             Eigen::DenseShape,
                                             Eigen::GemvProduct>
    : generic_product_impl_base<SecUtility::Math::LinearOperatorBase<Derived>,
                                Rhs,
                                generic_product_impl<SecUtility::Math::LinearOperatorBase<Derived>, Rhs>>
{
	/// <summary>
	/// Calculates <code>dst += alpha * lhs * rhs</code>, where <code>rhs</code> being a vector,
	/// <code>alpha</code> and <code>lhs</code> being scalar and matrix respectively.
	/// </summary>
	template <typename Dest>
	static void scaleAndAddTo(
	        Dest& dst,
	        const SecUtility::Math::LinearOperatorBase<Derived>& lhs,
	        const Rhs& rhs,
	        const std::common_type_t<typename traits<Derived>::Scalar, typename traits<Rhs>::Scalar>& alpha)
	{
		dst.noalias() += alpha * lhs.ApplyOn(rhs);
	}
};


template <typename Derived>
class SecUtility::Math::LinearOperatorBase : public Eigen::EigenBase<LinearOperatorBase<Derived>>
{
	using Base = Eigen::EigenBase<LinearOperatorBase>;

public:
	// DEV_NOTE: following are required typedefs, constants, and method
	using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
	using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
	using StorageIndex = Eigen::Index;

	static constexpr int ColsAtCompileTime = Eigen::Dynamic;
	static constexpr int RowsAtCompileTime = Eigen::Dynamic;
	static constexpr int MaxColsAtCompileTime = Eigen::Dynamic;
	static constexpr int MaxRowsAtCompileTime = Eigen::Dynamic;
	static constexpr int IsRowMajor = false;
	static constexpr int Flags = 0;

	/* CRTP VIRTUAL */ constexpr Eigen::Index rows() const noexcept
	{
		return static_cast<const Derived&>(*this).rows();
	}

	/* CRTP VIRTUAL */ constexpr Eigen::Index cols() const noexcept
	{
		return static_cast<const Derived&>(*this).cols();
	}

	template <typename Vector>
	/* CRTP VIRTUAL */ constexpr auto ApplyOn(const Vector& vector) const
	        noexcept(noexcept(static_cast<const Derived&>(*this).ApplyOn_Impl(vector)))
	{
		eigen_assert(vector.rows() == 1 || vector.cols() == 1);
		eigen_assert(vector.size() == cols());
		return static_cast<const Derived&>(*this).ApplyOn_Impl(vector);
	}

	template <typename DestScalar = Scalar>
	/* CRTP VIRTUAL */ Eigen::VectorX<DestScalar> ExtractColumn(const Eigen::Index col) const
	{
		static_assert(std::is_convertible_v<Scalar, DestScalar>);
		eigen_assert(cols() != Eigen::Dynamic);  // can't extract a column if we don't know the size of a column
		eigen_assert(col >= 0 && col < cols());
		Eigen::VectorX<DestScalar> v = Eigen::VectorX<DestScalar>::Zero(cols());
		v(col) = static_cast<DestScalar>(1);
		return ApplyOn(v);
	}

	template <typename DestScalar = Scalar>
	/* CRTP VIRTUAL */ Eigen::MatrixX<DestScalar> ToDense() const
	{
		static_assert(std::is_convertible_v<Scalar, DestScalar>);
		eigen_assert(rows() != Eigen::Dynamic && cols() != Eigen::Dynamic);
		Eigen::MatrixX<DestScalar> dense(rows(), cols());
		Eigen::VectorX<Scalar> v(cols());

		for (Eigen::Index i = 0; i < cols(); i++)
		{
			v.setZero();
			v(i) = static_cast<Scalar>(1);
			dense.col(i) = ApplyOn(v);
		}

		return dense;
	}

	template <typename Rhs>
	Eigen::Product<LinearOperatorBase, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const
	{
		return {*this, x.derived()};
	}
};
