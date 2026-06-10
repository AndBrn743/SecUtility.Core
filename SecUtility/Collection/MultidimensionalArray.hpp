// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>
#include <array>
#include <cstddef>
#include <functional>
#include <iterator>
#include <utility>
#include <vector>


namespace SecUtility
{
	namespace Detail::MultidimensionalArray
	{
		// Builds a constexpr array of per-dimension strides (row-major).
		// For shape {D0, D1, D2}: strides = {D1*D2, D2, 1}
		template <std::size_t... Dims>
		constexpr std::array<std::size_t, sizeof...(Dims)> MakeStrides() noexcept
		{
			constexpr std::size_t N = sizeof...(Dims);
			std::array<std::size_t, N> dims = {Dims...};
			std::array<std::size_t, N> strides{};
			strides[N - 1] = 1;
			for (std::size_t i = N - 1; i-- > 0;)
			{
				strides[i] = strides[i + 1] * dims[i + 1];
			}
			return strides;
		}

		template <std::size_t N>
		constexpr std::size_t ComputeFlatIndex(const std::array<std::size_t, N>& strides,
		                                       const std::array<std::size_t, N>& shape,
		                                       const std::array<std::size_t, N>& indices) noexcept
		{
			std::size_t flat = 0;
			for (std::size_t i = 0; i < N; ++i)
			{
				assert(indices[i] < shape[i] && "MultidimensionalArray: index out of range");
				flat += indices[i] * strides[i];
			}
			return flat;
		}

		template <std::size_t N, typename... Indices>
		constexpr std::array<std::size_t, N> PackIndicesIntoArray(Indices... idxs)
		{
			static_assert(sizeof...(Indices) == N, "Wrong number of indices");
			return {static_cast<std::size_t>(idxs)...};
		}
	}


	// -----------------------------------------------------------------------------
	// MultidimensionalArray<T, Dimensions...>
	//
	// A fully stack-allocated, row-major N-dimensional array.
	//
	// Example:
	//   MultidimensionalArray<float, 3, 4, 5> arr; // 3 * 4 * 5 array of floats
	//   arr(1, 2, 3) = 42.f;
	// -----------------------------------------------------------------------------
	template <typename T, std::size_t... Dimensions>
	class MultidimensionalArray
	{
		static_assert(sizeof...(Dimensions) > 0, "Must have at least one dimension");
		static_assert(((Dimensions > 0) && ...), "All dimensions must be non-zero");

	public:
		using value_type = T;
		using reference = T&;
		using const_reference = const T&;
		using pointer = T*;
		using const_pointer = const T*;
		using size_type = std::size_t;
		using iterator = pointer;
		using const_iterator = const_pointer;

		static constexpr size_type Rank = sizeof...(Dimensions);
		static constexpr size_type TotalSize = (Dimensions * ...);
		static constexpr std::array<size_type, Rank> Shape = {Dimensions...};
		static constexpr std::array<size_type, Rank> Strides =
		        Detail::MultidimensionalArray::MakeStrides<Dimensions...>();

		constexpr MultidimensionalArray() : m_Data{}
		{
			/* NO CODE */
		}

		explicit constexpr MultidimensionalArray(const T& fillValue)
		{
			m_Data.fill(fillValue);
		}

		// From flat initializer list  {1, 2, 3, 4, ...}
		// Note: std::initializer_list constructors cannot be fully constexpr in C++17
		// (std::copy is not constexpr until C++20). Use the variadic helper for constexpr.
		MultidimensionalArray(std::initializer_list<T> il) : m_Data{}
		{
			if (il.size() != TotalSize)
			{
				throw InvalidArgumentException("Initializer list size mismatch");
			}
			std::copy(il.begin(), il.end(), m_Data.begin());
		}

		template <typename... Values>
		static constexpr std::enable_if_t<sizeof...(Values) == TotalSize && (std::is_convertible_v<Values, T> && ...),
		                                  MultidimensionalArray>
		FromValues(Values&&... values)
		{
			MultidimensionalArray arr{};
			arr.m_Data = {static_cast<T>(std::forward<Values>(values))...};
			return arr;
		}

		template <typename... Indices>
		constexpr reference operator()(Indices... indices)
		{
			static_assert(sizeof...(Indices) == Rank, "Wrong number of indices");
			return m_Data[ComputeFlatIndex(indices...)];
		}

		template <typename... Indices>
		constexpr const_reference operator()(Indices... indices) const
		{
			static_assert(sizeof...(Indices) == Rank, "Wrong number of indices");
			if (((static_cast<std::size_t>(indices) >= Dimensions) || ...))
			{
				throw ArgumentOutOfRangeException("MultidimensionalArray: index out of range");
			}

			return m_Data[ComputeFlatIndex(indices...)];
		}

		template <typename... Indices>
		reference At(Indices... indices)
		{
			static_assert(sizeof...(Indices) == Rank, "Wrong number of indices");
			if (((static_cast<std::size_t>(indices) >= Dimensions) || ...))
			{
				throw ArgumentOutOfRangeException("MultidimensionalArray: index out of range");
			}

			return m_Data.at(ComputeFlatIndex(indices...));
		}

		template <typename... Indices>
		const_reference At(Indices... indices) const
		{
			static_assert(sizeof...(Indices) == Rank, "Wrong number of indices");
			return m_Data.at(ComputeFlatIndex(indices...));
		}

		// Raw flat access (no bounds check).
		constexpr reference Flat(const size_type i) noexcept
		{
			return m_Data[i];
		}

		// Raw flat access (no bounds check).
		constexpr const_reference Flat(const size_type i) const noexcept
		{
			return m_Data[i];
		}

		static constexpr size_type Size() noexcept
		{
			return TotalSize;
		}

		constexpr size_type Extent(const size_type dimension) const noexcept
		{
			return Shape[dimension];
		}

		static constexpr bool IsEmpty() noexcept
		{
			return TotalSize == 0;
		}

		constexpr pointer data() noexcept
		{
			return m_Data.data();
		}
		constexpr const_pointer data() const noexcept
		{
			return m_Data.data();
		}

		constexpr iterator begin() noexcept
		{
			return m_Data.begin();
		}

		constexpr iterator end() noexcept
		{
			return m_Data.end();
		}

		constexpr const_iterator begin() const noexcept
		{
			return m_Data.begin();
		}

		constexpr const_iterator end() const noexcept
		{
			return m_Data.end();
		}

		constexpr const_iterator cbegin() const noexcept
		{
			return m_Data.cbegin();
		}

		constexpr const_iterator cend() const noexcept
		{
			return m_Data.cend();
		}

		constexpr void Fill(const T& value)
		{
			m_Data.fill(value);
		}

		template <typename UnaryFunc>
		void Foreach(UnaryFunc&& f)
		{
			for (auto& v : m_Data)
			{
				f(v);
			}
		}

		template <typename UnaryFunc>
		void Foreach(UnaryFunc&& f) const
		{
			for (const auto& v : m_Data)
			{
				f(v);
			}
		}

		constexpr bool operator==(const MultidimensionalArray& other) const noexcept
		{
			return m_Data == other.m_Data;
		}

		constexpr bool operator!=(const MultidimensionalArray& other) const noexcept
		{
			return m_Data != other.m_Data;
		}

		// // -- Slice: fix the first dimension, return a (rank-1) view copy ----------
		// // e.g. arr.slice<1>() on a [3][4][5] returns a MultidimensionalArray<T,4,5>
		// // (Only defined when rank > 1.)
		// template <std::size_t I, std::size_t R = Rank, std::enable_if_t<(R > 1), int> = 0>
		// auto Slice(size_type firstIndex) const
		// {
		// 	return Slice_Impl<I>(firstIndex, std::make_index_sequence<Rank - 1>{});
		// }

	private:
		std::array<T, TotalSize> m_Data;

		template <typename... Indices>
		static constexpr size_type ComputeFlatIndex(Indices... indices)
		{
			const auto indexArray = Detail::MultidimensionalArray::PackIndicesIntoArray<Rank>(indices...);
			return Detail::MultidimensionalArray::ComputeFlatIndex<Rank>(Strides, Shape, indexArray);
		}

		// // Slice helper: builds the sub-array type by removing dimension 0.
		// template <std::size_t Fixed, std::size_t... Rest>
		// auto Slice_Impl(size_type firstIndex, std::index_sequence<Rest...>) const
		// {
		// 	static_assert(Fixed < Shape[0], "Slice index out of range");
		// 	// Sub-shape = Dimensions[1], Dimensions[2], ...
		// 	using SubArray = MultidimensionalArray<T, Shape[Rest + 1]...>;
		// 	SubArray sub;
		// 	const size_type base = firstIndex * Strides[0];
		// 	for (size_type i = 0; i < SubArray::TotalSize; ++i)
		// 	{
		// 		sub.Flat(i) = m_Data[base + i];
		// 	}
		// 	return sub;
		// }
	};
}
