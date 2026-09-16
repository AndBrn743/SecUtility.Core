// SPDX-License-Identifier: MPL-2.0
// Copyright (c) 2026 Andy Brown

#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


namespace Hoppy
{
	template <typename Derived>
	class LowRankMatrixBase;

	template <typename Scalar_, int RowsAtCompileTime_, int ColsAtCompileTime_>
	class LowRankMatrix;

	struct LowRankStorage
	{
	};

	template <typename Scalar>
	using LowRankMatrixX = LowRankMatrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
}


template <typename Scalar_, int RowsAtCompileTime_, int ColsAtCompileTime_>
struct Eigen::internal::traits<Hoppy::LowRankMatrix<Scalar_, RowsAtCompileTime_, ColsAtCompileTime_>>
{
	using Scalar = Scalar_;
	using StorageKind = Hoppy::LowRankStorage;
	using StorageIndex = Eigen::Index;
	using XprKind = Eigen::MatrixXpr;

	static constexpr int RowsAtCompileTime = RowsAtCompileTime_;
	static constexpr int ColsAtCompileTime = ColsAtCompileTime_;
	static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime_;
	static constexpr int MaxColsAtCompileTime = ColsAtCompileTime_;
	static constexpr int Flags = 0;
};


template <typename Derived>
class Hoppy::LowRankMatrixBase : public Eigen::EigenBase<Derived>
{
public:
	using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
	using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
	using StorageIndex = typename Eigen::internal::traits<Derived>::StorageIndex;

	static constexpr int RowsAtCompileTime = Eigen::internal::traits<Derived>::RowsAtCompileTime;
	static constexpr int ColsAtCompileTime = Eigen::internal::traits<Derived>::ColsAtCompileTime;
	static constexpr int MaxRowsAtCompileTime = Eigen::internal::traits<Derived>::MaxRowsAtCompileTime;
	static constexpr int MaxColsAtCompileTime = Eigen::internal::traits<Derived>::MaxColsAtCompileTime;
	static constexpr int IsRowMajor = false;
	static constexpr int Flags = Eigen::internal::traits<Derived>::Flags;

	[[nodiscard]] constexpr Eigen::Index rows() const noexcept { return asDerived().rowsImpl(); }
	[[nodiscard]] constexpr Eigen::Index cols() const noexcept { return asDerived().colsImpl(); }
	[[nodiscard]] Eigen::Index termCount() const noexcept { return asDerived().termCountImpl(); }

	[[nodiscard]] decltype(auto) coefficients() const noexcept { return asDerived().coefficientsImpl(); }
	[[nodiscard]] decltype(auto) leftVectors() const noexcept { return asDerived().leftVectorsImpl(); }
	[[nodiscard]] decltype(auto) rightVectors() const noexcept { return asDerived().rightVectorsImpl(); }

	[[nodiscard]] decltype(auto) coefficientOfTerm(const Eigen::Index index) const
	{
		return asDerived().coefficientOfTermImpl(index);
	}

	[[nodiscard]] decltype(auto) leftVectorOfTerm(const Eigen::Index index) const
	{
		return asDerived().leftVectorOfTermImpl(index);
	}

	[[nodiscard]] decltype(auto) rightVectorOfTerm(const Eigen::Index index) const
	{
		return asDerived().rightVectorOfTermImpl(index);
	}

	[[nodiscard]] auto term(const Eigen::Index index) const
	{
		struct Term
		{
			decltype(std::declval<const LowRankMatrixBase&>().leftVectorOfTerm(index)) leftVector;
			decltype(std::declval<const LowRankMatrixBase&>().coefficientOfTerm(index)) coefficient;
			decltype(std::declval<const LowRankMatrixBase&>().rightVectorOfTerm(index)) rightVector;
		};

		return Term{leftVectorOfTerm(index), coefficientOfTerm(index), rightVectorOfTerm(index)};
	}

protected:
	constexpr LowRankMatrixBase() noexcept = default;
	LowRankMatrixBase(const LowRankMatrixBase&) = default;
	LowRankMatrixBase(LowRankMatrixBase&&) noexcept = default;
	LowRankMatrixBase& operator=(const LowRankMatrixBase&) = default;
	LowRankMatrixBase& operator=(LowRankMatrixBase&&) noexcept = default;
	~LowRankMatrixBase() = default;

private:
	[[nodiscard]] constexpr const Derived& asDerived() const noexcept { return static_cast<const Derived&>(*this); }
};


namespace Hoppy::Detail
{
	inline std::size_t CheckedBufferSize(const Eigen::Index dimension, const std::size_t termCount)
	{
		eigen_assert(dimension >= 0 && "A matrix dimension cannot be negative");

		const auto unsignedDimension = static_cast<std::size_t>(dimension);
		if (unsignedDimension != 0 && termCount > std::numeric_limits<std::size_t>::max() / unsignedDimension)
		{
			throw std::length_error("Low-rank matrix storage size overflow");
		}
		return unsignedDimension * termCount;
	}
}


template <typename Scalar_, int RowsAtCompileTime_, int ColsAtCompileTime_>
class Hoppy::LowRankMatrix
    : public LowRankMatrixBase<LowRankMatrix<Scalar_, RowsAtCompileTime_, ColsAtCompileTime_>>
{
	static_assert(RowsAtCompileTime_ == Eigen::Dynamic || RowsAtCompileTime_ >= 0);
	static_assert(ColsAtCompileTime_ == Eigen::Dynamic || ColsAtCompileTime_ >= 0);

public:
	using Base = LowRankMatrixBase<LowRankMatrix>;
	friend Base;
	using Base::cols;
	using Base::rows;
	using Base::termCount;

	using Scalar = Scalar_;
	using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
	using StorageIndex = Eigen::Index;
	using CoefficientVector = Eigen::VectorX<Scalar>;
	using LeftVector = Eigen::Vector<Scalar, RowsAtCompileTime_>;
	using RightVector = Eigen::Vector<Scalar, ColsAtCompileTime_>;
	using LeftVectors = Eigen::Matrix<Scalar, RowsAtCompileTime_, Eigen::Dynamic>;
	using RightVectors = Eigen::Matrix<Scalar, ColsAtCompileTime_, Eigen::Dynamic>;

	static constexpr int RowsAtCompileTime = RowsAtCompileTime_;
	static constexpr int ColsAtCompileTime = ColsAtCompileTime_;
	static constexpr int MaxRowsAtCompileTime = RowsAtCompileTime_;
	static constexpr int MaxColsAtCompileTime = ColsAtCompileTime_;
	static constexpr int IsRowMajor = false;
	static constexpr int Flags = 0;

	constexpr LowRankMatrix() noexcept = default;
	LowRankMatrix(const LowRankMatrix&) = default;
	LowRankMatrix(LowRankMatrix&&) noexcept = default;
	LowRankMatrix& operator=(const LowRankMatrix&) = default;
	LowRankMatrix& operator=(LowRankMatrix&&) noexcept = default;
	~LowRankMatrix() = default;

	explicit LowRankMatrix(const Eigen::Index rows, const Eigen::Index cols) : m_Rows(rows), m_Cols(cols)
	{
		eigen_assert(rows >= 0 && cols >= 0);
	}

	template <typename LeftDerived, typename RightDerived>
	LowRankMatrix(const Scalar& coefficient,
	              const Eigen::MatrixBase<LeftDerived>& leftVector,
	              const Eigen::MatrixBase<RightDerived>& rightVector)
	    : m_Rows(leftVector.size()), m_Cols(rightVector.size())
	{
		addTerm(coefficient, leftVector, rightVector);
	}

	template <typename CoefficientsDerived, typename LeftDerived, typename RightDerived>
	LowRankMatrix(const Eigen::MatrixBase<CoefficientsDerived>& coefficients,
	              const Eigen::MatrixBase<LeftDerived>& leftVectors,
	              const Eigen::MatrixBase<RightDerived>& rightVectors)
	    : m_Rows(leftVectors.rows()), m_Cols(rightVectors.rows())
	{
		addTerms(coefficients, leftVectors, rightVectors);
	}

	[[nodiscard]] std::size_t capacity() const noexcept { return m_Coefficients.capacity(); }

	void reserve(const Eigen::Index termCapacity)
	{
		eigen_assert(termCapacity >= 0 && "A term capacity cannot be negative");
		const auto capacity = static_cast<std::size_t>(termCapacity);
		m_Coefficients.reserve(capacity);
		m_LeftVectorBuffer.reserve(Detail::CheckedBufferSize(rows(), capacity));
		m_RightVectorBuffer.reserve(Detail::CheckedBufferSize(cols(), capacity));
	}

	void clear() noexcept
	{
		m_Coefficients.clear();
		m_LeftVectorBuffer.clear();
		m_RightVectorBuffer.clear();
	}

	template <typename LeftDerived, typename RightDerived>
	LowRankMatrix& addTerm(const Scalar& coefficient,
	                       const Eigen::MatrixBase<LeftDerived>& leftVector,
	                       const Eigen::MatrixBase<RightDerived>& rightVector)
	{
		if (coefficient == Scalar{})
		{
			return *this;
		}

		validateVector(leftVector, rows());
		validateVector(rightVector, cols());
		const Scalar evaluatedCoefficient = coefficient;
		const Eigen::VectorX<Scalar> evaluatedLeft = leftVector.reshaped();
		const Eigen::VectorX<Scalar> evaluatedRight = rightVector.reshaped();
		appendEvaluatedTerm(evaluatedCoefficient, evaluatedLeft, evaluatedRight);
		return *this;
	}

	template <typename CoefficientsDerived, typename LeftDerived, typename RightDerived>
	LowRankMatrix& addTerms(const Eigen::MatrixBase<CoefficientsDerived>& coefficients,
	                        const Eigen::MatrixBase<LeftDerived>& leftVectors,
	                        const Eigen::MatrixBase<RightDerived>& rightVectors)
	{
		eigen_assert((coefficients.rows() == 1 || coefficients.cols() == 1)
		             && "Low-rank coefficients must be a vector");
		eigen_assert(coefficients.size() == leftVectors.cols() && coefficients.size() == rightVectors.cols()
		             && "Low-rank coefficient and vector term counts do not agree");
		eigen_assert(leftVectors.rows() == rows() && rightVectors.rows() == cols()
		             && "Low-rank term vectors do not match the matrix dimensions");

		const CoefficientVector evaluatedCoefficients = coefficients.reshaped();
		Eigen::Index nonzeroCount = 0;
		for (Eigen::Index index = 0; index < evaluatedCoefficients.size(); index++)
		{
			nonzeroCount += evaluatedCoefficients[index] == Scalar{} ? 0 : 1;
		}
		if (nonzeroCount == 0)
		{
			return *this;
		}

		CoefficientVector filteredCoefficients(nonzeroCount);
		Eigen::MatrixX<Scalar> filteredLeft(rows(), nonzeroCount);
		Eigen::MatrixX<Scalar> filteredRight(cols(), nonzeroCount);
		Eigen::Index destinationIndex = 0;
		for (Eigen::Index sourceIndex = 0; sourceIndex < evaluatedCoefficients.size(); sourceIndex++)
		{
			if (evaluatedCoefficients[sourceIndex] != Scalar{})
			{
				filteredCoefficients[destinationIndex] = evaluatedCoefficients[sourceIndex];
				filteredLeft.col(destinationIndex) = leftVectors.col(sourceIndex);
				filteredRight.col(destinationIndex) = rightVectors.col(sourceIndex);
				destinationIndex++;
			}
		}

		const auto finalCount = static_cast<std::size_t>(termCount() + nonzeroCount);
		reserve(static_cast<Eigen::Index>(finalCount));
		m_Coefficients.insert(m_Coefficients.end(), filteredCoefficients.data(),
		                      filteredCoefficients.data() + filteredCoefficients.size());
		appendBuffer(m_LeftVectorBuffer, filteredLeft.data(), filteredLeft.size());
		appendBuffer(m_RightVectorBuffer, filteredRight.data(), filteredRight.size());
		return *this;
	}

private:
	[[nodiscard]] constexpr Eigen::Index rowsImpl() const noexcept { return m_Rows.value(); }
	[[nodiscard]] constexpr Eigen::Index colsImpl() const noexcept { return m_Cols.value(); }
	[[nodiscard]] Eigen::Index termCountImpl() const noexcept
	{
		return static_cast<Eigen::Index>(m_Coefficients.size());
	}

	/// Read-only views into owned storage. Any operation that increases capacity invalidates all existing views.
	/// `clear` invalidates term views and references but retains the buffers and their capacity.
	[[nodiscard]] auto coefficientsImpl() const noexcept
	{
		return Eigen::Map<const CoefficientVector>{m_Coefficients.data(), termCount()};
	}

	[[nodiscard]] auto leftVectorsImpl() const noexcept
	{
		return Eigen::Map<const LeftVectors>{m_LeftVectorBuffer.data(), rows(), termCount()};
	}

	[[nodiscard]] auto rightVectorsImpl() const noexcept
	{
		return Eigen::Map<const RightVectors>{m_RightVectorBuffer.data(), cols(), termCount()};
	}

	[[nodiscard]] const Scalar& coefficientOfTermImpl(const Eigen::Index index) const
	{
		validateTermIndex(index);
		return m_Coefficients[static_cast<std::size_t>(index)];
	}

	[[nodiscard]] auto leftVectorOfTermImpl(const Eigen::Index index) const
	{
		validateTermIndex(index);
		return Eigen::Map<const LeftVector>{m_LeftVectorBuffer.data() + rows() * index, rows()};
	}

	[[nodiscard]] auto rightVectorOfTermImpl(const Eigen::Index index) const
	{
		validateTermIndex(index);
		return Eigen::Map<const RightVector>{m_RightVectorBuffer.data() + cols() * index, cols()};
	}

	template <typename Derived>
	static void validateVector(const Eigen::MatrixBase<Derived>& vector, const Eigen::Index expectedSize)
	{
		eigen_assert((vector.rows() == 1 || vector.cols() == 1) && vector.size() == expectedSize
		             && "A low-rank term vector has the wrong shape or dimension");
	}

	void validateTermIndex(const Eigen::Index index) const
	{
		eigen_assert(index >= 0 && index < termCount() && "Low-rank term index is out of range");
	}

	void appendEvaluatedTerm(const Scalar& coefficient,
	                         const Eigen::VectorX<Scalar>& leftVector,
	                         const Eigen::VectorX<Scalar>& rightVector)
	{
		reserve(termCount() + 1);
		m_Coefficients.push_back(coefficient);
		appendBuffer(m_LeftVectorBuffer, leftVector.data(), leftVector.size());
		appendBuffer(m_RightVectorBuffer, rightVector.data(), rightVector.size());
	}

	static void appendBuffer(std::vector<Scalar>& destination, const Scalar* const source, const Eigen::Index size)
	{
		if (size != 0)
		{
			destination.insert(destination.end(), source, source + size);
		}
	}

private:
	Eigen::internal::variable_if_dynamic<Eigen::Index, RowsAtCompileTime> m_Rows{};
	Eigen::internal::variable_if_dynamic<Eigen::Index, ColsAtCompileTime> m_Cols{};
	std::vector<Scalar> m_Coefficients{};
	std::vector<Scalar> m_LeftVectorBuffer{};
	std::vector<Scalar> m_RightVectorBuffer{};
};
