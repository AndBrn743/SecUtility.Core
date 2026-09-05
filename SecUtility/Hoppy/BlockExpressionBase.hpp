// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>
#include <SecUtility/Collection/SubscriptBasedIterator.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>
#include <cmath>
#include <ostream>

namespace Hoppy
{
	namespace Detail
	{
		template <typename T, typename = void>
		struct has_data : std::false_type
		{};

		template <typename T>
		struct has_data<T, std::void_t<decltype(std::declval<const T&>().data())>> : std::true_type
		{};

		template <typename Source, typename Destination>
		void assertNoOverlap(const Source& source, const Destination& destination)
		{
			if constexpr (has_data<Source>::value && has_data<Destination>::value)
			{
				if (source.storedSize() == 0 || destination.size() == 0)
					return;
				const auto sourceBegin = reinterpret_cast<std::uintptr_t>(source.data());
				const auto sourceEnd = sourceBegin + static_cast<std::uintptr_t>(source.storedSize()) * sizeof(typename Source::Scalar);
				const auto destinationBegin = reinterpret_cast<std::uintptr_t>(destination.data());
				const auto destinationEnd = destinationBegin + static_cast<std::uintptr_t>(destination.size()) * sizeof(typename Destination::Scalar);
				eigen_assert(destinationEnd <= sourceBegin || sourceEnd <= destinationBegin);
			}
		}
	}  // namespace Detail

	template <typename Derived>
	class BlockExpressionBase : public Eigen::EigenBase<Derived>
	{
	public:
		using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
		using CoeffReturnType = Scalar;
		using StorageIndex = Eigen::Index;
		using Iterator = SecUtility::SubscriptBasedIterator<Derived, Eigen::Index>;
		using ConstIterator = SecUtility::SubscriptBasedIterator<const Derived, Eigen::Index>;
		static constexpr int SizeAtCompileTime = Eigen::Dynamic;
		static constexpr int MaxSizeAtCompileTime = Eigen::Dynamic;
		static constexpr bool IsVectorAtCompileTime =
		        Eigen::internal::traits<Derived>::RowsAtCompileTime == 1
		        || Eigen::internal::traits<Derived>::ColsAtCompileTime == 1;

		using Eigen::EigenBase<Derived>::const_cast_derived;
		using Eigen::EigenBase<Derived>::derived;

		Eigen::Index blockCount() const { return derived().blockCount(); }
		Eigen::Index rows() const { return derived().rows(); }
		Eigen::Index cols() const { return derived().cols(); }
		Eigen::Index size() const { return derived().size(); }
		Eigen::Index totalDimension() const { return derived().totalDimension(); }
		Eigen::Index storedSize() const { return derived().storedSize(); }
		Eigen::Index dimensionOfBlock(Eigen::Index index) const { return derived().dimensionOfBlock(index); }
		Eigen::Index blockOffset(Eigen::Index index) const { return derived().blockOffset(index); }
		Eigen::Index storageOffset(Eigen::Index index) const { return derived().storageOffset(index); }
		const std::vector<Eigen::Index>& blockingInfo() const& { return derived().blockingInfo(); }
		std::vector<Eigen::Index> blockingInfo() const&&
		{
			return std::move(derived()).blockingInfo();
		}

		template <typename TOther>
		bool hasSameBlockingAs(const TOther& other) const { return blockingInfo() == other.blockingInfo(); }

		Iterator begin() { return Iterator(derived(), 0); }
		Iterator end() { return Iterator(derived(), blockCount()); }
		ConstIterator begin() const { return cbegin(); }
		ConstIterator end() const { return cend(); }
		ConstIterator cbegin() const { return ConstIterator(derived(), 0); }
		ConstIterator cend() const { return ConstIterator(derived(), blockCount()); }

		Derived& setConstant(const Scalar& value)
		{
			for (Eigen::Index i = 0; i < blockCount(); ++i)
				derived()[i].setConstant(value);
			return derived();
		}
		Derived& setZero() { return setConstant(Scalar{}); }
		Derived& setOnes() { return setConstant(Scalar{1}); }
		Derived& setRandom() { for (Eigen::Index i = 0; i < blockCount(); ++i) derived()[i].setRandom(); return derived(); }

		Scalar sum() const { Scalar result{}; for (Eigen::Index i = 0; i < blockCount(); ++i) result += derived()[i].sum(); return result; }
		using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
		RealScalar squaredNorm() const { RealScalar result{}; for (Eigen::Index i = 0; i < blockCount(); ++i) result += derived()[i].squaredNorm(); return result; }
		RealScalar norm() const { using std::sqrt; return sqrt(squaredNorm()); }
		RealScalar rootMeanSquare() const { if (size() == 0) return RealScalar{}; using std::sqrt; return sqrt(squaredNorm() / static_cast<RealScalar>(size())); }
		Scalar mean() const { if (size() == 0) { eigen_assert(false && "mean requires a nonempty expression"); return Scalar{}; } return sum() / static_cast<Scalar>(size()); }
		RealScalar maxAbsCoeff() const
		{
			if (storedSize() == 0) { eigen_assert(false && "maxAbsCoeff requires a nonempty expression"); return RealScalar{}; }
			RealScalar result = derived()[0].cwiseAbs().maxCoeff();
			for (Eigen::Index i = 1; i < blockCount(); ++i) result = (std::max)(result, derived()[i].cwiseAbs().maxCoeff());
			return result;
		}
		template <typename S = Scalar, typename = std::enable_if_t<!Eigen::NumTraits<S>::IsComplex>>
		Scalar maxCoeff() const
		{
			if (storedSize() == 0) { eigen_assert(false && "maxCoeff requires a nonempty expression"); return Scalar{}; }
			Scalar result = derived()[0].maxCoeff();
			for (Eigen::Index i = 1; i < blockCount(); ++i) result = (std::max)(result, derived()[i].maxCoeff());
			return result;
		}
		template <typename S = Scalar, typename = std::enable_if_t<!Eigen::NumTraits<S>::IsComplex>>
		Scalar minCoeff() const
		{
			if (storedSize() == 0) { eigen_assert(false && "minCoeff requires a nonempty expression"); return Scalar{}; }
			Scalar result = derived()[0].minCoeff();
			for (Eigen::Index i = 1; i < blockCount(); ++i) result = (std::min)(result, derived()[i].minCoeff());
			return result;
		}
		bool allFinite() const { for (Eigen::Index i = 0; i < blockCount(); ++i) if (!derived()[i].allFinite()) return false; return true; }
		bool hasNaN() const { for (Eigen::Index i = 0; i < blockCount(); ++i) if (derived()[i].hasNaN()) return true; return false; }
		template <typename Other>
		bool isApprox(const Other& other, const RealScalar& precision = Eigen::NumTraits<Scalar>::dummy_precision()) const
		{
			if (!hasSameBlockingAs(other)) { eigen_assert(false && "isApprox requires equal blocking"); return false; }
			for (Eigen::Index i = 0; i < blockCount(); ++i) if (!derived()[i].isApprox(other[i], precision)) return false;
			return true;
		}
	};

	template <typename Derived>
	std::ostream& operator<<(std::ostream& stream, const BlockExpressionBase<Derived>& expression)
	{
		for (Eigen::Index index = 0; index < expression.blockCount(); ++index)
		{
			if (index != 0)
				stream << '\n' << '\n';
			stream << expression.derived()[index];
		}
		return stream;
	}
}  // namespace Hoppy
