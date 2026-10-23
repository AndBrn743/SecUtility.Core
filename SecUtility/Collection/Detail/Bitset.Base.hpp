// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if !defined(SEC_BITSET_DETAIL)
#error "Please do not directly include internal headers"
#endif

#include <SecUtility/Collection/SubscriptBasedIterator.hpp>
#include <SecUtility/IO/BitOrder.hpp>
#include <SecUtility/Macro/ConstevalIf.hpp>
#include <SecUtility/Macro/ForceInline.hpp>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <ostream>
#include <string>
#include <vector>

#if !defined(SEC_ASSERT)
#define SEC_ASSERT(expr) assert(expr)
#endif
#if !defined(SEC_NOEXCEPT)
#define SEC_NOEXCEPT noexcept
#endif

namespace SecUtility
{
	template <typename Derived>
	class BitsetBase
	{
	private:
		// ----------------------------------------------------------
		//  CRTP plumbing (private, friend Derived only)
		// ----------------------------------------------------------
		friend Derived;

		template <typename>
		friend class BitsetBase;

		template <typename>
		friend class WordAlignedBitsetBase;

		template <typename>
		friend class OwningBitsetBase;

		template <typename>
		friend class Detail::Bitset::BitsetSegmentExpr;

		template <typename>
		friend class Detail::Bitset::BitsetNotExpr;

		template <typename, typename, typename>
		friend class Detail::Bitset::BitsetBinaryExpr;

		constexpr BitsetBase() noexcept = default;
		constexpr BitsetBase(const BitsetBase&) noexcept = default;
		constexpr BitsetBase(BitsetBase&&) noexcept = default;
		constexpr BitsetBase& operator=(const BitsetBase&) noexcept = default;
		constexpr BitsetBase& operator=(BitsetBase&&) noexcept = default;
		~BitsetBase() noexcept = default;

		SEC_FORCE_INLINE constexpr std::size_t BlockCount() const noexcept
		{
			return Detail::Bitset::BlocksFor(AsDerived().HeadPaddingBitCount() + AsDerived().Size());
		}

		/* CRTP VIRTUAL */ SEC_FORCE_INLINE constexpr std::uint64_t& Block(const std::size_t index) noexcept
		{
			return AsDerived().Block(index);
		}

		/* CRTP VIRTUAL */ SEC_FORCE_INLINE constexpr std::uint64_t Block(const std::size_t index) const noexcept
		{
			return AsDerived().Block(index);
		}


		/* CRTP VIRTUAL */ SEC_FORCE_INLINE constexpr std::size_t HeadPaddingBitCount() const
		        SEC_NOEXCEPT  // corresponds to trailing
		{
			const auto padding = AsDerived().HeadPaddingBitCount();
			SEC_ASSERT(padding < Detail::Bitset::BitsPerBlock);
			return padding;
		}

		constexpr std::size_t TailPaddingBitCount() const SEC_NOEXCEPT  // corresponds to leading
		{
			const auto padding = BlockCount() * Detail::Bitset::BitsPerBlock - (Size() + HeadPaddingBitCount());
			SEC_ASSERT(padding < Detail::Bitset::BitsPerBlock);
			return padding;
		}

		SEC_FORCE_INLINE constexpr std::uint64_t MaskOfBlock(const std::size_t index) const SEC_NOEXCEPT
		{
			return AsDerived().MaskOfBlock_Impl(index);
		}

		/* CRTP VIRTUAL */ SEC_FORCE_INLINE constexpr std::uint64_t MaskOfBlock_Impl(const std::size_t index) const
		        SEC_NOEXCEPT
		{
			if (BlockCount() == 1)
			{
				return Detail::Bitset::HeadMask(HeadPaddingBitCount())
				       & Detail::Bitset::TailMask(HeadPaddingBitCount() + Size());
			}

			if (index == 0)
			{
				return Detail::Bitset::HeadMask(HeadPaddingBitCount());
			}

			if (index + 1 == BlockCount())
			{
				return Detail::Bitset::TailMask(HeadPaddingBitCount() + Size());
			}

			return ~std::uint64_t{0};
		}

		constexpr auto MakeBlockMaskCaches() const SEC_NOEXCEPT
		{
			return AsDerived().MakeBlockMaskCaches_Impl();
		}

		/* CRTP VIRTUAL */ constexpr auto MakeBlockMaskCaches_Impl() const SEC_NOEXCEPT
		{
			const auto head = MaskOfBlock(0);
			const auto backIndex = BlockCount() - 1;
			const auto tail = backIndex == 0 ? head : MaskOfBlock(backIndex);
			return [head, tail, backIndex](const std::size_t index)
			{
				if (index == 0)
				{
					return head;
				}
				if (index == backIndex)
				{
					return tail;
				}
				else
				{
					return ~std::uint64_t{0};
				}
			};
		}

		constexpr decltype(auto) AlignedTo(const std::size_t headPadding) const SEC_NOEXCEPT
		{
			SEC_ASSERT(headPadding < Detail::Bitset::BitsPerBlock);
			return AsDerived().AlignedTo_Impl(headPadding);
		}

		/* CRTP VIRTUAL */ constexpr Derived AlignedTo_Impl(std::size_t headPadding) const noexcept = delete;

		constexpr void RestPaddingBitsOfZerothBlock() noexcept
		{
			if (const std::size_t size = Size(); size != 0)
			{
				Block(0) &= Detail::Bitset::HeadMask(HeadPaddingBitCount());
			}
		}

		constexpr void RestPaddingBitsOfLastBlock() noexcept
		{
			if (const std::size_t paddedSize = Size() + HeadPaddingBitCount(); paddedSize != 0)
			{
				Block(BlockCount() - 1) &= Detail::Bitset::TailMask(paddedSize);
			}
		}


		// Word i of the logical bit sequence, with no head padding. The caller
		// supplies a valid word index and masks unused bits in the final word.
		constexpr std::uint64_t UnpaddedBlock(const std::size_t i) const SEC_NOEXCEPT
		{
			return AsDerived().UnpaddedBlock_Impl(i);
		}

		/* CRTP VIRTUAL */ constexpr std::uint64_t UnpaddedBlock_Impl(const std::size_t i) const SEC_NOEXCEPT
		{
			return ShiftedBlock(-static_cast<int>(HeadPaddingBitCount()), i);
		}

		constexpr std::uint64_t ShiftedBlock(const int shift, const std::size_t i) const SEC_NOEXCEPT
		{
			if (const std::size_t count = BlockCount(); i >= count)
			{
				// Left realignment can add a block containing only carry bits.
				// All other reads beyond the source contribute zero.
				if (shift > 0 && i == count && count != 0)
				{
					return Block(count - 1) >> (Detail::Bitset::BitsPerBlock - static_cast<std::size_t>(shift));
				}
				return 0;
			}
			if (shift == 0)
			{
				return Block(i);
			}
			if (shift > 0)
			{
				// Shift left: low bits come from Block(i), high bits come from Block(i-1)
				const auto lshift = static_cast<std::size_t>(shift);
				const std::uint64_t lo = Block(i) << lshift;
				const std::uint64_t hi =
				        i > 0 ? Block(i - 1) >> (Detail::Bitset::BitsPerBlock - lshift) : std::uint64_t{0};
				return hi | lo;
			}
			else
			{
				// Shift right: high bits come from Block(i), low bits come from Block(i+1)
				const auto rshift = static_cast<std::size_t>(-shift);
				const std::uint64_t hi = Block(i) >> rshift;
				const std::uint64_t lo = i + 1 < BlockCount() ? Block(i + 1) << (Detail::Bitset::BitsPerBlock - rshift)
				                                              : std::uint64_t{0};
				return hi | lo;
			}
		}

		// just a cast to `typename Traits<Derived>::EvaluatedType`. template is use here for delay the instantiation
		template <typename..., typename D = Derived>
		constexpr typename Traits<D>::EvaluatedType EvalHelper() const
		        noexcept(noexcept(static_cast<typename Traits<D>::EvaluatedType>(AsDerived())))
		{
			return static_cast<typename Traits<D>::EvaluatedType>(AsDerived());
		}

	public:
		// ----------------------------------------------------------
		//  Proxy reference
		// ----------------------------------------------------------
		class BitReference
		{
			friend BitsetBase;
			constexpr BitReference(BitsetBase& derived, const std::size_t index) noexcept
			    : r_Derived(derived.AsDerived()), m_PaddedIndex(index + r_Derived.HeadPaddingBitCount()),
			      m_Mask(std::uint64_t{1} << (m_PaddedIndex % Detail::Bitset::BitsPerBlock))
			{
				/* NO CODE */
			}

		public:
			constexpr BitReference& operator=(const bool value) noexcept
			{
				if (value)
				{
					r_Derived.Block(m_PaddedIndex / Detail::Bitset::BitsPerBlock) |= m_Mask;
				}
				else
				{
					r_Derived.Block(m_PaddedIndex / Detail::Bitset::BitsPerBlock) &= ~m_Mask;
				}
				return *this;
			}

			// ReSharper disable once CppNonExplicitConversionOperator
			constexpr operator bool() const noexcept
			{
				return static_cast<bool>(std::as_const(r_Derived).Block(m_PaddedIndex / Detail::Bitset::BitsPerBlock)
				                         & m_Mask);
			}

			// ReSharper disable once CppMemberFunctionMayBeConst
			constexpr void Flip() noexcept
			{
				r_Derived.Block(m_PaddedIndex / Detail::Bitset::BitsPerBlock) ^= m_Mask;
			}

		private:
			Derived& r_Derived;
			std::size_t m_PaddedIndex;
			std::uint64_t m_Mask;
		};

	public:
		constexpr const Derived& AsDerived() const noexcept
		{
			return static_cast<const Derived&>(*this);
		}

		constexpr Derived& AsDerived() noexcept
		{
			return static_cast<Derived&>(*this);
		}

		// ----------------------------------------------------------
		//  Conversion from other derived
		// ----------------------------------------------------------
		template <typename OtherDerived>
		constexpr Derived& operator=(const BitsetBase<OtherDerived>& other) SEC_NOEXCEPT
		{
			SEC_ASSERT(Size() == other.Size());
			AsDerived().Assign_Impl(other);
			return AsDerived();
		}

		// ----------------------------------------------------------
		//  Size / capacity
		// ----------------------------------------------------------
		SEC_FORCE_INLINE constexpr std::size_t Size() const noexcept
		{
			return AsDerived().Size();
		}

		// ----------------------------------------------------------
		//  Element access
		// ----------------------------------------------------------
		constexpr bool operator[](const std::size_t index) const SEC_NOEXCEPT
		{
			SEC_ASSERT(index < Size());
			return (Block((index + HeadPaddingBitCount()) / Detail::Bitset::BitsPerBlock)
			        >> ((index + HeadPaddingBitCount()) % Detail::Bitset::BitsPerBlock))
			       & 1u;
		}

		constexpr BitReference operator[](const std::size_t index) SEC_NOEXCEPT
		{
			SEC_ASSERT(index < Size());
			return {*this, index};
		}

		constexpr bool IsOne(const std::size_t index) const SEC_NOEXCEPT
		{
			return (*this)[index];
		}

		constexpr bool IsZero(const std::size_t index) const SEC_NOEXCEPT
		{
			return !(*this)[index];
		}

		constexpr Derived& Set(const std::size_t index, const bool value = true) SEC_NOEXCEPT
		{
			SEC_ASSERT(index < Size());
			(*this)[index] = value;
			return AsDerived();
		}

		constexpr Derived& Reset(const std::size_t index) SEC_NOEXCEPT
		{
			SEC_ASSERT(index < Size());
			(*this)[index] = false;
			return AsDerived();
		}

		constexpr Derived& Flip(const std::size_t index) SEC_NOEXCEPT
		{
			SEC_ASSERT(index < Size());
			(*this)[index].Flip();
			return AsDerived();
		}

		// ----------------------------------------------------------
		//  Iterator
		// ----------------------------------------------------------
		constexpr auto begin() noexcept
		{
			return SubscriptBasedIterator<Derived>{AsDerived(), 0};
		}

		constexpr auto begin() const noexcept
		{
			return SubscriptBasedIterator<const Derived>{AsDerived(), 0};
		}

		constexpr auto cbegin() const noexcept  // NOLINT
		{
			return SubscriptBasedIterator<const Derived>{AsDerived(), 0};
		}

		constexpr auto end() noexcept
		{
			return SubscriptBasedIterator<Derived>{AsDerived(), Size()};
		}

		constexpr auto end() const noexcept
		{
			return SubscriptBasedIterator<const Derived>{AsDerived(), Size()};
		}

		constexpr auto cend() const noexcept  // NOLINT
		{
			return SubscriptBasedIterator<const Derived>{AsDerived(), Size()};
		}

		constexpr auto rbegin() noexcept  // NOLINT
		{
			return std::reverse_iterator{end()};
		}

		constexpr auto rbegin() const noexcept  // NOLINT
		{
			return std::reverse_iterator{end()};
		}

		constexpr auto crbegin() const noexcept  // NOLINT
		{
			return std::reverse_iterator{cend()};
		}

		constexpr auto rend() noexcept
		{
			return std::reverse_iterator{begin()};
		}

		constexpr auto rend() const noexcept
		{
			return std::reverse_iterator{begin()};
		}

		constexpr auto crend() const noexcept  // NOLINT
		{
			return std::reverse_iterator{cbegin()};
		}


	public:
		constexpr Detail::Bitset::BitsetSegmentExpr<Derived> Segment(const std::size_t start,
		                                                             const std::size_t count) noexcept
		{
			return {AsDerived(), start, count};
		}

		constexpr Detail::Bitset::BitsetSegmentExpr<const Derived> Segment(const std::size_t start,
		                                                                   const std::size_t count) const noexcept
		{
			return {AsDerived(), start, count};
		}

		constexpr Detail::Bitset::BitsetSegmentExpr<Derived> Leading(const std::size_t count) noexcept
		{
			return {AsDerived(), Size() - count, count};
		}

		constexpr Detail::Bitset::BitsetSegmentExpr<const Derived> Leading(const std::size_t count) const noexcept
		{
			return {AsDerived(), Size() - count, count};
		}

		constexpr Detail::Bitset::BitsetSegmentExpr<Derived> Trailing(const std::size_t count) noexcept
		{
			return {AsDerived(), 0, count};
		}

		constexpr Detail::Bitset::BitsetSegmentExpr<const Derived> Trailing(const std::size_t count) const noexcept
		{
			return {AsDerived(), 0, count};
		}


		// ----------------------------------------------------------
		//  Whole-set operations
		// ----------------------------------------------------------
		constexpr void SetAll(const bool value = true) SEC_NOEXCEPT
		{
			AsDerived().SetAll_Impl(value);
		}

		constexpr void ResetAll() noexcept
		{
			SetAll(false);
		}

		constexpr void FlipAll() noexcept
		{
			AsDerived().FlipAll_Impl();
		}

		constexpr bool IsAllOnes() const noexcept
		{
			if (Size() == 0)
			{
				return false;
			}
			return AsDerived().IsAllOnes_Impl();
		}

		constexpr bool IsAllZeros() const noexcept
		{
			if (Size() == 0)
			{
				return false;
			}
			return AsDerived().IsAllZeros_Impl();
		}

		constexpr bool HasOnes() const noexcept
		{
			return Size() != 0 && !IsAllZeros();
		}

		constexpr bool HasZeros() const noexcept
		{
			return Size() != 0 && !IsAllOnes();
		}

		// ----------------------------------------------------------
		//  Counting
		// ----------------------------------------------------------
		constexpr std::size_t OneCount() const noexcept
		{
			return AsDerived().OneCount_Impl();
		}

		constexpr std::size_t ZeroCount() const noexcept
		{
			return Size() - OneCount();
		}

		constexpr std::size_t TrailingZeroCount() const noexcept
		{
			if (Size() == 0)
			{
				return 0;
			}

			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				if (const std::uint64_t block = Traits<Derived>::IsMaskedForRead ? Block(i) : Block(i) & MaskOfBlock(i);
				    block != 0)
				{
					return i * Detail::Bitset::BitsPerBlock + Detail::Bitset::TrailingZeroCount(block)
					       - HeadPaddingBitCount();
				}
			}

			return Size();
		}

		constexpr std::size_t TrailingOneCount() const noexcept
		{
			return (~AsDerived()).TrailingZeroCount();
		}

		constexpr std::size_t LeadingZeroCount() const noexcept
		{
			if (Size() == 0)
			{
				return 0;
			}

			for (std::size_t i = BlockCount(); i-- > 0;)
			{
				if (const std::uint64_t block = Traits<Derived>::IsMaskedForRead ? Block(i) : Block(i) & MaskOfBlock(i);
				    block != 0)
				{
					return (BlockCount() - 1 - i) * Detail::Bitset::BitsPerBlock
					       + Detail::Bitset::LeadingZeroCount(block) - TailPaddingBitCount();
				}
			}

			return Size();
		}

		constexpr std::size_t LeadingOneCount() const noexcept
		{
			return (~AsDerived()).LeadingZeroCount();
		}


		// ----------------------------------------------------------
		//  Index-of queries  (return Size() when not found)
		// ----------------------------------------------------------
		constexpr std::size_t IndexOfFirstOne() const noexcept
		{
			return TrailingZeroCount();
		}

		constexpr std::size_t IndexOfFirstZero() const noexcept
		{
			return TrailingOneCount();
		}

		constexpr std::size_t IndexOfLastOne() const noexcept
		{
			const std::size_t lzc = LeadingZeroCount();
			return lzc == Size() ? Size() : Size() - 1 - lzc;
		}

		constexpr std::size_t IndexOfLastZero() const noexcept
		{
			const std::size_t loc = LeadingOneCount();
			return loc == Size() ? Size() : Size() - 1 - loc;
		}

		constexpr std::size_t IndexOfNextOne(const std::size_t pos) const SEC_NOEXCEPT
		{
			if (pos + 1 >= Size())
			{
				return Size();
			}

			const std::size_t startBit = HeadPaddingBitCount() + pos + 1;
			std::size_t blockIdx = startBit / Detail::Bitset::BitsPerBlock;
			const std::size_t bitInBlk = startBit % Detail::Bitset::BitsPerBlock;

			if (const std::uint64_t blk = Block(blockIdx) >> bitInBlk; blk != 0)
			{
				const std::size_t idx = startBit + Detail::Bitset::TrailingZeroCount(blk) - HeadPaddingBitCount();
				// guards against over-counting caused by padding-bits leak into payload-bits segment
				return idx < Size() ? idx : Size();
			}

			for (++blockIdx; blockIdx < BlockCount(); ++blockIdx)
			{
				if (const std::uint64_t blk = Block(blockIdx); blk != 0)
				{
					const std::size_t idx = blockIdx * Detail::Bitset::BitsPerBlock
					                        + Detail::Bitset::TrailingZeroCount(blk) - HeadPaddingBitCount();
					// guards against over-counting caused by padding-bits leak into payload-bits segment
					return idx < Size() ? idx : Size();
				}
			}
			return Size();
		}

		constexpr std::size_t IndexOfNextZero(const std::size_t pos) const SEC_NOEXCEPT
		{
			return (~AsDerived()).IndexOfNextOne(pos);
		}

		constexpr std::size_t IndexOfPreviousOne(const std::size_t pos) const SEC_NOEXCEPT
		{
			if (pos == 0)
			{
				return Size();
			}

			const std::size_t endBit = HeadPaddingBitCount() + pos - 1;
			std::size_t blockIdx = endBit / Detail::Bitset::BitsPerBlock;
			const std::size_t bitInBlk = endBit % Detail::Bitset::BitsPerBlock;

			if (const std::uint64_t blk = Block(blockIdx) << (Detail::Bitset::BitsPerBlock - bitInBlk - 1); blk != 0)
			{
				const auto idx = blockIdx * Detail::Bitset::BitsPerBlock + bitInBlk
				                 - Detail::Bitset::LeadingZeroCount(blk) - HeadPaddingBitCount();
				// guards against over-counting caused by padding-bits leak into payload-bits segment
				return idx < Size() ? idx : Size();
			}

			while (blockIdx-- > 0)
			{
				if (const std::uint64_t blk = Block(blockIdx); blk != 0)
				{
					const auto idx = blockIdx * Detail::Bitset::BitsPerBlock
					                 + (Detail::Bitset::BitsPerBlock - 1 - Detail::Bitset::LeadingZeroCount(blk))
					                 - HeadPaddingBitCount();
					// guards against over-counting caused by padding-bits leak into payload-bits segment
					return idx < Size() ? idx : Size();
				}
			}
			return Size();
		}

		constexpr std::size_t IndexOfPreviousZero(const std::size_t pos) const SEC_NOEXCEPT
		{
			return (~AsDerived()).IndexOfPreviousOne(pos);
		}


#if false
		// ----------------------------------------------------------
		//  Shift operations  (bit 0 == LSB of Data()[0])
		//  <<  shifts bits toward higher indices ("left" in math notation)
		//  >>  shifts bits toward lower  indices
		// ----------------------------------------------------------
		LeftShiftExpr<const Derived> operator<<(std::size_t shift) const;

		RightShiftExpr<const Derived> operator>>(std::size_t shift) const;
#endif

		// ----------------------------------------------------------
		//  Bitwise operations
		// ----------------------------------------------------------
		template <typename OtherDerived>
		Derived& operator&=(const BitsetBase<OtherDerived>& other) SEC_NOEXCEPT
		{
			return BitwiseCombinationOp(other, std::bit_and<>{});
		}

		template <typename OtherDerived>
		Derived& operator|=(const BitsetBase<OtherDerived>& other) SEC_NOEXCEPT
		{
			return BitwiseCombinationOp(other, std::bit_or<>{});
		}

		template <typename OtherDerived>
		Derived& operator^=(const BitsetBase<OtherDerived>& other) SEC_NOEXCEPT
		{
			return BitwiseCombinationOp(other, std::bit_xor<>{});
		}

		template <typename OtherDerived>
		auto operator&(const BitsetBase<OtherDerived>& other) const SEC_NOEXCEPT
		{
			return Detail::Bitset::BitsetBinaryExpr<std::bit_and<>, Derived, OtherDerived>(AsDerived(),
			                                                                               other.AsDerived());
		}

		template <typename OtherDerived>
		auto operator|(const BitsetBase<OtherDerived>& other) const SEC_NOEXCEPT
		{
			return Detail::Bitset::BitsetBinaryExpr<std::bit_or<>, Derived, OtherDerived>(AsDerived(),
			                                                                              other.AsDerived());
		}

		template <typename OtherDerived>
		auto operator^(const BitsetBase<OtherDerived>& other) const SEC_NOEXCEPT
		{
			return Detail::Bitset::BitsetBinaryExpr<std::bit_xor<>, Derived, OtherDerived>(AsDerived(),
			                                                                               other.AsDerived());
		}

		auto operator~() const noexcept
		{
			return Detail::Bitset::BitsetNotExpr<const Derived>{AsDerived()};
		}

		constexpr Derived& operator<<=(const std::size_t n) SEC_NOEXCEPT
		{
			if (n == 0)
			{
				return AsDerived();
			}

			if (n >= Size())
			{
				ResetAll();
				return AsDerived();
			}

			AsDerived().ShiftLeft_Impl(n);
			return AsDerived();
		}

		constexpr Derived& operator>>=(const std::size_t n) SEC_NOEXCEPT
		{
			if (n == 0)
			{
				return AsDerived();
			}

			if (n >= Size())
			{
				ResetAll();
				return AsDerived();
			}

			AsDerived().ShiftRight_Impl(n);
			return AsDerived();
		}

		// ----------------------------------------------------------
		//  Comparison
		// ----------------------------------------------------------
		template <typename OtherDerived>
		constexpr bool operator==(const BitsetBase<OtherDerived>& other) const SEC_NOEXCEPT
		{
			if (Size() != other.Size())
			{
				return false;
			}

			return AsDerived().Equal_Impl(other);
		}

		template <typename OtherDerived>
		constexpr bool operator!=(const BitsetBase<OtherDerived>& other) const SEC_NOEXCEPT
		{
			return !(*this == other.AsDerived());
		}

		// ----------------------------------------------------------
		//  Conversion
		// ----------------------------------------------------------
		// the conventional ordering have leftmost character at highest index bit
		std::string ToString(const bool isUsingConventionalBitOrdering = true) const
		{
			const std::size_t size = Size();
			std::string s(size, '-');

			if (isUsingConventionalBitOrdering)
			{
				std::transform(cbegin(), cend(), s.rbegin(), [](const bool b) { return b ? '1' : '0'; });
			}
			else
			{
				std::transform(cbegin(), cend(), s.begin(), [](const bool b) { return b ? '1' : '0'; });
			}

			return s;
		}

		// use the conventional bit ordering by default. one could switch the behavior with LsbFirst and MsbFirst.
		// e.g., `os << LsbFirst << bs;`
		friend std::ostream& operator<<(std::ostream& os, BitsetBase& bs)
		{
			return os << bs.ToString(os.iword(BitOrderFlag()) == 0);
		}

		auto Eval() const
		{
			return EvalHelper();
		}


	private:
		// Default kernels for views and expressions. Owning storage supplies specialized hooks.
		template <typename OtherDerived>
		constexpr void Assign_Impl(const BitsetBase<OtherDerived>& other) SEC_NOEXCEPT
		{
			BitwiseCombinationOp(other, [](std::uint64_t, const std::uint64_t rhs) { return rhs; });
		}

		// Merge active bits while preserving the storage outside a view.
		// Owning destinations use this fallback when their source is a view or expression.
		static constexpr std::uint64_t MergeMaskedBlock(const std::uint64_t original,
		                                                const std::uint64_t value,
		                                                const std::uint64_t mask) noexcept
		{
			if constexpr (Traits<Derived>::IsMaskedForRead)
			{
				return value & mask;
			}
			else
			{
				return (original & ~mask) | (value & mask);
			}
		}

		constexpr void SetAll_Impl(const bool value) SEC_NOEXCEPT
		{
			const std::size_t count = BlockCount();
			const auto masks = MakeBlockMaskCaches();
			const std::uint64_t fill = value ? ~std::uint64_t{0} : std::uint64_t{0};
			for (std::size_t i = 0; i < count; ++i)
			{
				Block(i) = MergeMaskedBlock(Block(i), fill, masks(i));
			}
		}

		constexpr void FlipAll_Impl() noexcept
		{
			const std::size_t count = BlockCount();
			const auto masks = MakeBlockMaskCaches();
			for (std::size_t i = 0; i < count; ++i)
			{
				Block(i) = MergeMaskedBlock(Block(i), ~Block(i), masks(i));
			}
		}

		constexpr std::size_t OneCount_Impl() const noexcept
		{
			std::size_t count = 0;
			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				count += Traits<Derived>::IsMaskedForRead ? Detail::Bitset::PopCount(Block(i))
				                                          : Detail::Bitset::PopCount(MaskOfBlock(i) & Block(i));
			}
			return count;
		}

		constexpr bool IsAllOnes_Impl() const noexcept
		{
			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				if (const auto mask = MaskOfBlock(i); (mask & Block(i)) != mask)
				{
					return false;
				}
			}
			return true;
		}

		constexpr bool IsAllZeros_Impl() const noexcept
		{
			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				const auto block = MaskOfBlock(i) & Block(i);
				if (block != 0)
				{
					return false;
				}
			}

			return true;
		}

		constexpr void ShiftLeft_Impl(const std::size_t n) SEC_NOEXCEPT
		{
			const std::size_t blockShift = n / Detail::Bitset::BitsPerBlock;
			const std::size_t bitShift = n % Detail::Bitset::BitsPerBlock;
			const std::size_t count = BlockCount();
			const std::size_t last = count - 1;
			const auto masks = MakeBlockMaskCaches();

			// Move high words first so in-place reads precede their overwrites.
			if (bitShift == 0)
			{
				Block(last) = MergeMaskedBlock(Block(last), Block(last - blockShift), masks(last));
				for (std::size_t i = last; i-- > blockShift;)
				{
					Block(i) = Block(i - blockShift);
				}
			}
			else
			{
				if (last > blockShift)
				{
					const std::size_t source = last - blockShift;
					const std::uint64_t value = (Block(source) << bitShift)
					                            | (Block(source - 1) >> (Detail::Bitset::BitsPerBlock - bitShift));
					Block(last) = MergeMaskedBlock(Block(last), value, masks(last));
				}
				for (std::size_t i = last; i-- > blockShift + 1;)
				{
					const std::size_t source = i - blockShift;
					Block(i) = (Block(source) << bitShift)
					           | (Block(source - 1) >> (Detail::Bitset::BitsPerBlock - bitShift));
				}
				Block(blockShift) = MergeMaskedBlock(Block(blockShift), Block(0) << bitShift, masks(blockShift));
			}

			// This also clears bits shifted in from the storage before the view.
			Trailing(n).ResetAll();
		}

		constexpr void ShiftRight_Impl(const std::size_t n) SEC_NOEXCEPT
		{
			const std::size_t blockShift = n / Detail::Bitset::BitsPerBlock;
			const std::size_t bitShift = n % Detail::Bitset::BitsPerBlock;
			const std::size_t count = BlockCount();
			const std::size_t last = count - blockShift - 1;
			const auto masks = MakeBlockMaskCaches();

			// Move low words first so in-place reads precede their overwrites.
			if (bitShift == 0)
			{
				Block(0) = MergeMaskedBlock(Block(0), Block(blockShift), masks(0));
				for (std::size_t i = 1; i <= last; ++i)
				{
					Block(i) = Block(i + blockShift);
				}
			}
			else
			{
				if (last != 0)
				{
					const std::uint64_t value = (Block(blockShift) >> bitShift)
					                            | (Block(blockShift + 1) << (Detail::Bitset::BitsPerBlock - bitShift));
					Block(0) = MergeMaskedBlock(Block(0), value, masks(0));
				}
				for (std::size_t i = 1; i < last; ++i)
				{
					const std::size_t source = i + blockShift;
					Block(i) = (Block(source) >> bitShift)
					           | (Block(source + 1) << (Detail::Bitset::BitsPerBlock - bitShift));
				}
				Block(last) = MergeMaskedBlock(Block(last), Block(count - 1) >> bitShift, masks(last));
			}

			// This also clears bits shifted in from the storage after the view.
			Leading(n).ResetAll();
		}

		template <typename OtherDerived>
		constexpr bool Equal_Impl(const BitsetBase<OtherDerived>& other) const SEC_NOEXCEPT
		{
			return ExecuteBinaryOpByAlignment(
			        *this,
			        other,
			        &BitsetBase::OperatorEqualEqual_AssumeSameSizeAndAligned<OtherDerived>,
			        &BitsetBase::OperatorEqualEqual_AssumeSameSizeAndMisaligned<OtherDerived>);
		}

		template <typename OtherDerived, typename Op>
		Derived& BitwiseCombinationOp_Impl(const BitsetBase<OtherDerived>& other, const Op op) SEC_NOEXCEPT
		{
			return ExecuteBinaryOpByAlignment(*this,
			                                  other,
			                                  &BitsetBase::BitwiseCombinationOp_AssumeAligned<OtherDerived, Op>,
			                                  &BitsetBase::BitwiseCombinationOp_AssumeMisaligned<OtherDerived, Op>,
			                                  op);
		}

		template <typename Self, typename OtherDerived, typename AlignedOp, typename MisalignedOp, typename... Args>
		static decltype(auto) ExecuteBinaryOpByAlignment(Self&& self,
		                                                 const BitsetBase<OtherDerived>& other,
		                                                 AlignedOp&& alignedOp,
		                                                 MisalignedOp&& misalignedOp,
		                                                 Args&&... args) SEC_NOEXCEPT
		{
			SEC_ASSERT(self.Size() == other.Size());

			if (self.HeadPaddingBitCount() == other.HeadPaddingBitCount())
			{
				return std::invoke(alignedOp, self.AsDerived(), other.AsDerived(), std::forward<Args>(args)...);
			}
			else if constexpr (Traits<Derived>::IsCheaplyRealignable && Traits<OtherDerived>::IsCheaplyRealignable)
			{
				if (self.BlockCount() > other.BlockCount())
				{
					return std::invoke(alignedOp,
					                   self.AlignedTo(other.HeadPaddingBitCount()),
					                   other.AsDerived(),
					                   std::forward<Args>(args)...);
				}
				else
				{
					return std::invoke(alignedOp,
					                   self.AsDerived(),
					                   other.AlignedTo(self.HeadPaddingBitCount()),
					                   std::forward<Args>(args)...);
				}
			}
			else if constexpr (Traits<OtherDerived>::IsCheaplyRealignable)
			{
				return std::invoke(alignedOp,
				                   self.AsDerived(),
				                   other.AlignedTo(self.HeadPaddingBitCount()),
				                   std::forward<Args>(args)...);
			}
			else if constexpr (Traits<Derived>::IsCheaplyRealignable)
			{
				return std::invoke(alignedOp,
				                   self.AlignedTo(other.HeadPaddingBitCount()),
				                   other.AsDerived(),
				                   std::forward<Args>(args)...);
			}
			else
			{
				return std::invoke(misalignedOp, self.AsDerived(), other.AsDerived(), std::forward<Args>(args)...);
			}
		}

		template <typename OtherDerived, typename Op>
		Derived& BitwiseCombinationOp(const BitsetBase<OtherDerived>& other, const Op op) SEC_NOEXCEPT
		{
			return AsDerived().BitwiseCombinationOp_Impl(other, op);
		}

		template <typename OtherDerived, typename Op>
		Derived& BitwiseCombinationOp_AssumeAligned(const BitsetBase<OtherDerived>& other,
		                                            const Op op = {}) SEC_NOEXCEPT
		{
			const auto masks = MakeBlockMaskCaches();
			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				const auto mask = masks(i);
				const auto b = Block(i);
				Block(i) = MergeMaskedBlock(b, op(b, other.Block(i)), mask);
			}

			return AsDerived();
		}

		template <typename OtherDerived, typename Op>
		Derived& BitwiseCombinationOp_AssumeMisaligned(const BitsetBase<OtherDerived>& other,
		                                               const Op op = {}) SEC_NOEXCEPT
		{
			SEC_ASSERT(Size() == other.Size());

			const auto masks = MakeBlockMaskCaches();

			const std::size_t thisHead = HeadPaddingBitCount();
			const std::size_t otherHead = other.HeadPaddingBitCount();

			// How many bits to shift other's raw data to align to *this.
			// Positive = shift left, negative = shift right.
			const std::ptrdiff_t shift = static_cast<std::ptrdiff_t>(thisHead) - static_cast<std::ptrdiff_t>(otherHead);
			SEC_ASSERT(shift != 0);

			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				const auto mask = masks(i);
				const auto current = Block(i);
				const auto source = other.ShiftedBlock(shift, i);
				const auto result = op(current, source);
				Block(i) = MergeMaskedBlock(current, result, mask);
			}

			return AsDerived();
		}

		template <typename OtherDerived>
		constexpr bool OperatorEqualEqual_AssumeSameSizeAndAligned(const BitsetBase<OtherDerived>& other) const
		        SEC_NOEXCEPT
		{
			const std::size_t count = BlockCount();
			SEC_ASSERT(count == other.BlockCount());
			const auto masks = MakeBlockMaskCaches();
			for (std::size_t i = 0; i < count; ++i)
			{
				const auto mask = masks(i);
				if ((Traits<Derived>::IsMaskedForRead ? Block(i) : Block(i) & mask)
				    != (Traits<OtherDerived>::IsMaskedForRead ? other.Block(i) : other.Block(i) & mask))
				{
					return false;
				}
			}
			return true;
		}

		template <typename OtherDerived>
		constexpr bool OperatorEqualEqual_AssumeSameSizeAndMisaligned(const BitsetBase<OtherDerived>& other) const
		        SEC_NOEXCEPT
		{
			const std::size_t size = Size();
			const std::size_t count = Detail::Bitset::BlocksFor(size);
			const std::uint64_t tail = Detail::Bitset::TailMask(size);
			const int leftShift = -static_cast<int>(HeadPaddingBitCount());
			const int rightShift = -static_cast<int>(other.HeadPaddingBitCount());
			for (std::size_t i = 0; i < count; ++i)
			{
				const auto mask = i + 1 == count ? tail : ~std::uint64_t{0};
				if ((ShiftedBlock(leftShift, i) & mask) != (other.ShiftedBlock(rightShift, i) & mask))
				{
					return false;
				}
			}
			return true;
		}
	};
}
