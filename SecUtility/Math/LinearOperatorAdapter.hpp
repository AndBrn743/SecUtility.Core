// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if __cplusplus < 202002L
#error "LinearOperatorAdapter requires C++20 or later"
#endif

#include <Eigen/Core>

#include <concepts>
#include <type_traits>
#include <utility>


namespace SecUtility::Math
{
	namespace Detail
	{
		template <typename Matrix>
		using LinearOperatorSource = std::remove_cvref_t<Matrix>;

		template <typename Matrix>
		concept AdaptableLinearOperator = requires(const LinearOperatorSource<Matrix>& matrix,
		                                           const Eigen::MatrixX<typename LinearOperatorSource<Matrix>::Scalar>&
		                                                   vectors) {
			typename LinearOperatorSource<Matrix>::Scalar;
			{ matrix.rows() } -> std::convertible_to<Eigen::Index>;
			{ matrix.cols() } -> std::convertible_to<Eigen::Index>;
			{ (matrix * vectors).eval() };
		};

		template <typename Matrix>
		concept AdaptableSelfAdjointLinearOperator = AdaptableLinearOperator<Matrix>
		                                             && requires(const LinearOperatorSource<Matrix>& matrix) {
			                                             { matrix.diagonal().real() };
		                                             };

		template <typename Matrix>
		[[nodiscard]] decltype(auto) NestLinearOperatorSource(Matrix&& matrix)
		{
			using Source = std::remove_cvref_t<Matrix>;
			if constexpr (std::is_lvalue_reference_v<Matrix&&>)
			{
				return std::forward<Matrix>(matrix);
			}
			else if constexpr (requires { typename Source::PlainObject; })
			{
				if constexpr (std::is_same_v<Source, typename Source::PlainObject>)
				{
					return Source{std::forward<Matrix>(matrix)};
				}
				else
				{
					return matrix.eval();
				}
			}
			else
			{
				return Source{std::forward<Matrix>(matrix)};
			}
		}
	}


	/// Adapts an Eigen or Hoppy matrix expression to the Core `rows`, `cols`, and `ApplyOn` protocol.
	/// Factory-created adapters reference lvalue sources and own rvalue sources. Rvalue Eigen expression trees are
	/// evaluated into an owning plain matrix so references held by the original tree cannot dangle.
	template <typename Nested_>
	class LinearOperatorAdapter
	{
	public:
		using Nested = Nested_;
		using Source = std::remove_cvref_t<Nested>;
		using Scalar = Source::Scalar;
		using RealScalar = Eigen::NumTraits<Scalar>::Real;

		explicit LinearOperatorAdapter(Nested nested) : m_Nested(std::forward<Nested>(nested)) {}

		[[nodiscard]] Eigen::Index rows() const noexcept { return m_Nested.rows(); }
		[[nodiscard]] Eigen::Index cols() const noexcept { return m_Nested.cols(); }

		template <typename Derived>
		[[nodiscard]] auto ApplyOn(const Eigen::MatrixBase<Derived>& vectors) const
		{
			eigen_assert(vectors.rows() == cols() && "Operator and right-hand side dimensions do not agree");
			return (m_Nested * vectors).eval();
		}

		[[nodiscard]] const Source& nestedExpression() const noexcept { return m_Nested; }

	private:
		Nested m_Nested;
	};


	/// Adds the real-valued `Diagonal` operation required by the Core self-adjoint linear-operator protocol.
	/// Construction asserts that the source is square but deliberately does not inspect every coefficient. Using this
	/// adapter is therefore an explicit caller assertion that the represented operator is self-adjoint.
	template <typename Nested_>
	class SelfAdjointLinearOperatorAdapter : public LinearOperatorAdapter<Nested_>
	{
		using Base = LinearOperatorAdapter<Nested_>;

	public:
		using typename Base::RealScalar;
		using typename Base::Scalar;
		using Base::ApplyOn;
		using Base::cols;
		using Base::rows;

		explicit SelfAdjointLinearOperatorAdapter(Nested_ nested) : Base(std::forward<Nested_>(nested))
		{
			eigen_assert(rows() == cols() && "A self-adjoint linear operator must be square");
		}

		[[nodiscard]] Eigen::VectorX<RealScalar> Diagonal() const
		{
			return Base::nestedExpression().diagonal().real();
		}
	};


	template <typename Matrix>
		requires Detail::AdaptableLinearOperator<Matrix>
	[[nodiscard]] auto MakeLinearOperatorAdapter(Matrix&& matrix)
	{
		using Nested = decltype(Detail::NestLinearOperatorSource(std::forward<Matrix>(matrix)));
		return LinearOperatorAdapter<Nested>{Detail::NestLinearOperatorSource(std::forward<Matrix>(matrix))};
	}


	template <typename Matrix>
		requires Detail::AdaptableSelfAdjointLinearOperator<Matrix>
	[[nodiscard]] auto MakeSelfAdjointLinearOperatorAdapter(Matrix&& matrix)
	{
		using Nested = decltype(Detail::NestLinearOperatorSource(std::forward<Matrix>(matrix)));
		return SelfAdjointLinearOperatorAdapter<Nested>{
		        Detail::NestLinearOperatorSource(std::forward<Matrix>(matrix))};
	}
}
