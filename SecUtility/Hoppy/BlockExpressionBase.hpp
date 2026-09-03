// SPDX-License-Identifier: MIT

#pragma once

#include <SecUtility/Hoppy/ForwardDeclarations.hpp>
#include <SecUtility/Collection/SubscriptBasedIterator.hpp>

#include <Eigen/Core>

#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

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
		std::vector<Eigen::Index> blockingInfo() const { return derived().blockingInfo(); }

		template <typename TOther>
		bool hasSameBlockingAs(const TOther& other) const { return blockingInfo() == other.blockingInfo(); }

		Iterator begin() { return Iterator(derived(), 0); }
		Iterator end() { return Iterator(derived(), blockCount()); }
		ConstIterator begin() const { return cbegin(); }
		ConstIterator end() const { return cend(); }
		ConstIterator cbegin() const { return ConstIterator(derived(), 0); }
		ConstIterator cend() const { return ConstIterator(derived(), blockCount()); }
	};
}  // namespace Hoppy
