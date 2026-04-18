// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <functional>
#include <type_traits>
#include <utility>

#include "IsScalar.hpp"
#include "LinearOperatorBase.hpp"


namespace SecUtility::Math
{
	namespace Detail::MatrixFreeLinearOperator
	{
		// Detects the scaler type of given function using the implicit type promotion rule.
		// It's a bit of trickery and I don't love it, but it's what I got for now.
		template <typename Functor>
		using ScalarOfFunctor = typename Eigen::internal::traits<
		        std::decay_t<decltype(std::declval<Functor>()(Eigen::VectorX<std::int8_t>{}))>>::Scalar;

		// Only to use std::enable_if_t with operator*, as we can't decltype lambdas in C++17,
		// and we don't have requires-clause in C++17 too
		template <typename TScalar, typename TNestedFunctor>
		struct ScaledFunctor
		{
			using Scalar = std::common_type_t<TScalar, ScalarOfFunctor<TNestedFunctor>>;
			TScalar ScalingFactor;
			TNestedFunctor NestedFunctor;

			template <typename Vector>
			Eigen::VectorX<Scalar> operator()(const Vector& vector) const
			{
				return ScalingFactor * NestedFunctor.operator()(vector);
			}
		};
	}

	/// \brief Matrix-free linear operator wrapper
	///
	/// Wraps a callable (lambda, function object, std::function, etc.) that represents a linear
	/// transformation, providing Eigen integration and operator composition capabilities.
	///
	/// \tparam Functor The callable type. Must be invocable with Eigen::VectorX<T> for some T.
	///
	/// Example usage:
	/// \code
	/// // Generic lambda
	/// MatrixFreeLinearOperator op1([](const auto& v) { return 2 * v; }, 5);
	///
	/// // Typed lambda
	/// MatrixFreeLinearOperator op2([](const Eigen::VectorX<double>& v) { return v; }, 5);
	///
	/// // Stateful functor
	/// int count = 0;
	/// auto op3 = MatrixFreeLinearOperator(
	///     [&count](const auto& v) mutable { ++count; return v; }, 5);
	/// \endcode
	///
	/// \warning The _get_functor() method is exposed only for internal use by arithmetic operators.
	///          It is not part of the public API and may change without notice.
	template <typename Functor>
	class MatrixFreeLinearOperator;
}


template <typename Functor>
struct Eigen::internal::traits<SecUtility::Math::MatrixFreeLinearOperator<Functor>>
    : traits<SparseMatrix<SecUtility::Math::Detail::MatrixFreeLinearOperator::ScalarOfFunctor<Functor>>>
{
	/* NO CODE */
};


template <typename Functor>
class SecUtility::Math::MatrixFreeLinearOperator : public LinearOperatorBase<MatrixFreeLinearOperator<Functor>>
{
public:
	using Base = LinearOperatorBase<MatrixFreeLinearOperator>;
	friend Base;

public:
	using Scalar = typename Base::Scalar;
	using NestedExpression = MatrixFreeLinearOperator;

	using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
	using StorageIndex = Eigen::Index;

	static constexpr int ColsAtCompileTime = Eigen::Dynamic;
	static constexpr int RowsAtCompileTime = Eigen::Dynamic;
	static constexpr int MaxColsAtCompileTime = Eigen::Dynamic;
	static constexpr int MaxRowsAtCompileTime = Eigen::Dynamic;
	static constexpr int IsRowMajor = false;
	static constexpr int Flags = 0;

	constexpr MatrixFreeLinearOperator() noexcept = default;

	constexpr MatrixFreeLinearOperator(const MatrixFreeLinearOperator& other)
	    : m_Functor(Functor{other.m_Functor}), m_Rows(other.m_Rows), m_Cols(other.m_Cols)
	{
		/* NO CODE */
	}

	constexpr MatrixFreeLinearOperator(MatrixFreeLinearOperator&& other) noexcept
	    : m_Functor(std::move(other.m_Functor)), m_Rows(other.m_Rows), m_Cols(other.m_Cols)
	{
		/* NO CODE */
	}

	constexpr MatrixFreeLinearOperator& operator=(const MatrixFreeLinearOperator& other)
	{
		if (this != &other)
		{
			m_Functor = Functor{other.m_Functor};
			m_Rows = other.m_Rows;
			m_Cols = other.m_Cols;
		}
		return *this;
	}

	constexpr MatrixFreeLinearOperator& operator=(MatrixFreeLinearOperator&& other) noexcept
	{
		if (this != &other)
		{
			m_Functor = std::move(other.m_Functor);
			m_Rows = other.m_Rows;
			m_Cols = other.m_Cols;
		}
		return *this;
	}

	~MatrixFreeLinearOperator() noexcept = default;

	constexpr MatrixFreeLinearOperator(Functor functor, const Eigen::Index rows, const Eigen::Index cols)
	    : m_Functor(std::move(functor)), m_Rows(rows), m_Cols(cols)
	{
		/* NO CODE */
	}

	constexpr MatrixFreeLinearOperator(Functor functor, const Eigen::Index dimension)
	    : m_Functor(std::move(functor)), m_Rows(dimension), m_Cols(dimension)
	{
		/* NO CODE */
	}

	/* CRTP OVERRIDE*/ constexpr Eigen::Index rows() const noexcept
	{
		return m_Rows;
	}

	/* CRTP OVERRIDE*/ constexpr Eigen::Index cols() const noexcept
	{
		return m_Cols;
	}

	/// INTERNAL USE ONLY: Required for operator implementation.
	/// Not part of the public API - subject to change without notice!
	/// Do not use in application code!
	constexpr const Functor& _get_functor() const noexcept
	{
		return m_Functor;
	}


private:
	template <typename Vector>
	constexpr auto ApplyOn_Impl(const Vector& vector) const
	{
		static_assert(std::is_base_of_v<Eigen::MatrixBase<std::decay_t<Vector>>, Vector>);
		eigen_assert(vector.rows() == 1 || vector.cols() == 1);
		return m_Functor(vector);
	}


private:
	Functor m_Functor{};
	Eigen::Index m_Rows{};
	Eigen::Index m_Cols{};
};


namespace SecUtility::Math
{

	template <typename Functor>
	constexpr const auto& operator+(const MatrixFreeLinearOperator<Functor>& operand)
	{
		return operand;
	}

	// we can't put the following functions inside the class definition. since each op will result in a new functor
	// type, the template type parameter of the result MatrixFreeLinearOperator will be determined with CTAD. With
	// in-class definition, template type parameter will be automatically filled to match the template definition itself
	// which is not we need in this case

	template <typename Functor>
	constexpr auto operator-(const MatrixFreeLinearOperator<Functor>& operand)
	{
		return MatrixFreeLinearOperator{
		        [f = Functor{operand._get_functor()}](const auto& vector)
		                -> Eigen::VectorX<typename Eigen::internal::traits<decltype(-f(vector))>::Scalar>
		        { return -f(vector); },
		        operand.rows(),
		        operand.cols()};
	}

	template <typename LhsFunctor, typename RhsFunctor>
	constexpr auto operator+(const MatrixFreeLinearOperator<LhsFunctor>& lhs,
	                         const MatrixFreeLinearOperator<RhsFunctor>& rhs)
	{
		eigen_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());

		return MatrixFreeLinearOperator{
		        [lf = LhsFunctor{lhs._get_functor()}, rf = RhsFunctor{rhs._get_functor()}](const auto& vector)
		                -> Eigen::VectorX<typename Eigen::internal::traits<decltype(lf(vector) + rf(vector))>::Scalar>
		        { return lf(vector) + rf(vector); },
		        lhs.rows(),
		        rhs.cols()};
	}

	template <typename LhsFunctor, typename RhsFunctor>
	constexpr auto operator-(const MatrixFreeLinearOperator<LhsFunctor>& lhs,
	                         const MatrixFreeLinearOperator<RhsFunctor>& rhs)
	{
		eigen_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());

		return MatrixFreeLinearOperator{
		        [lf = LhsFunctor{lhs._get_functor()}, rf = RhsFunctor{rhs._get_functor()}](const auto& vector)
		                -> Eigen::VectorX<typename Eigen::internal::traits<decltype(lf(vector) - rf(vector))>::Scalar>
		        { return lf(vector) - rf(vector); },
		        lhs.rows(),
		        rhs.cols()};
	}

	template <typename LhsFunctor, typename RhsFunctor>
	constexpr auto operator*(const MatrixFreeLinearOperator<LhsFunctor>& lhs,
	                         const MatrixFreeLinearOperator<RhsFunctor>& rhs)
	{
		eigen_assert(lhs.cols() == rhs.rows());

		return MatrixFreeLinearOperator{
		        [lf = LhsFunctor{lhs._get_functor()}, rf = RhsFunctor{rhs._get_functor()}](const auto& vector)
		                -> Eigen::VectorX<typename Eigen::internal::traits<decltype(lf(rf(vector)))>::Scalar>
		        { return lf(rf(vector)); },
		        lhs.rows(),
		        rhs.cols()};
	}

	template <typename Functor, typename OtherScalar>
	constexpr std::enable_if_t<
	        IsScalar<OtherScalar>,
	        MatrixFreeLinearOperator<Detail::MatrixFreeLinearOperator::ScaledFunctor<OtherScalar, Functor>>>
	operator*(const MatrixFreeLinearOperator<Functor>& linearOperator, const OtherScalar& scalar)
	{
		return {{scalar, Functor{linearOperator._get_functor()}}, linearOperator.rows(), linearOperator.cols()};
	}

	template <typename OtherScalar, typename Functor>
	constexpr std::enable_if_t<
	        IsScalar<OtherScalar>,
	        MatrixFreeLinearOperator<Detail::MatrixFreeLinearOperator::ScaledFunctor<OtherScalar, Functor>>>
	operator*(const OtherScalar& scalar, const MatrixFreeLinearOperator<Functor>& linearOperator)
	{
		return {{scalar, Functor{linearOperator._get_functor()}}, linearOperator.rows(), linearOperator.cols()};
	}

	template <typename LhsFunctor, typename OtherScalar>
	constexpr auto operator/(const MatrixFreeLinearOperator<LhsFunctor>& linearOperator, const OtherScalar& scalar)
	{
		static_assert(SecUtility::Math::IsScalar<std::decay_t<OtherScalar>>);

		return MatrixFreeLinearOperator{
		        [lf = LhsFunctor{linearOperator._get_functor()}, scalar](const auto& vector)
		                -> Eigen::VectorX<typename Eigen::internal::traits<decltype(lf(vector) / scalar)>::Scalar>
		        { return lf(vector) / scalar; },
		        linearOperator.rows(),
		        linearOperator.cols()};
	}
}
