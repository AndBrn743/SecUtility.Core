// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if !defined(SEC_BITSET_DETAIL)
#error "Please do not directly include internal headers"
#endif

#include <SecUtility/Collection/SubscriptBasedIterator.hpp>
#include <SecUtility/IO/BitOrder.hpp>
#include <SecUtility/Macro/ForceInline.hpp>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
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
			return Detail::Bitset::BlocksFor(AsDerived().HeadPadding() + AsDerived().Size());
		}

		/* CRTP VIRTUAL */ SEC_FORCE_INLINE constexpr std::uint64_t& Block(const std::size_t index) noexcept
		{
			return AsDerived().Block(index);
		}

		/* CRTP VIRTUAL */ SEC_FORCE_INLINE constexpr std::uint64_t Block(const std::size_t index) const noexcept
		{
			return AsDerived().Block(index);
		}

		/* CRTP VIRTUAL */ SEC_FORCE_INLINE constexpr std::size_t HeadPadding() const
		        SEC_NOEXCEPT  // corresponds to trailing
		{
			const auto padding = AsDerived().HeadPadding();
			SEC_ASSERT(padding < Detail::Bitset::BitsPerBlock);
			return padding;
		}

		constexpr std::size_t TailPadding() const SEC_NOEXCEPT  // corresponds to leading
		{
			const auto padding = BlockCount() * Detail::Bitset::BitsPerBlock - (Size() + HeadPadding());
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
				return (HeadPadding() == 0 ? ~std::uint64_t{0} : ~Detail::Bitset::LastBlockMask(HeadPadding()))
				       & Detail::Bitset::LastBlockMask(HeadPadding() + Size());
			}

			if (BlockCount() > 1 && index == 0)
			{
				return HeadPadding() == 0 ? ~std::uint64_t{0} : ~Detail::Bitset::LastBlockMask(HeadPadding());
			}

			if (BlockCount() > 1 && index + 1 == BlockCount() && TailPadding() != 0)
			{
				return Detail::Bitset::LastBlockMask(HeadPadding() + Size());
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

		constexpr decltype(auto) AlignedTo(const std::size_t headPadding) SEC_NOEXCEPT
		{
			SEC_ASSERT(headPadding < Detail::Bitset::BitsPerBlock);
			return AsDerived().AlignedTo_Impl(headPadding);
		}

		/* CRTP VIRTUAL */ constexpr Derived AlignedTo_Impl(const std::size_t headPadding) noexcept = delete;

		constexpr void RestPaddingBitsOfZerothBlock() noexcept
		{
			if (const std::size_t size = Size(); size != 0)
			{
				Block(0) &= ~Detail::Bitset::LastBlockMask(HeadPadding());
			}
		}

		constexpr void RestPaddingBitsOfLastBlock() noexcept
		{
			if (const std::size_t paddedSize = Size() + HeadPadding(); paddedSize != 0)
			{
				Block(BlockCount() - 1) &= Detail::Bitset::LastBlockMask(paddedSize);
			}
		}

		constexpr std::uint64_t ShiftedBlock(const int shift, const std::size_t i) const SEC_NOEXCEPT
		{
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
			    : r_Derived(derived.AsDerived()), m_PaddedIndex(index + r_Derived.HeadPadding()),
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
			BitwiseCombinationOp(other, [](std::uint64_t, const std::uint64_t rhs) { return rhs; });
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
			return (Block((index + HeadPadding()) / Detail::Bitset::BitsPerBlock)
			        >> ((index + HeadPadding()) % Detail::Bitset::BitsPerBlock))
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
			const auto masks = MakeBlockMaskCaches();
			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				if (value)
				{
					Block(i) |= masks(i);
				}
				else
				{
					Block(i) &= ~masks(i);
				}
			}
		}

		constexpr void ResetAll() noexcept
		{
			SetAll(false);
		}

		constexpr void FlipAll() noexcept
		{
			const auto masks = MakeBlockMaskCaches();
			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				const auto mask = masks(i);
				const auto block = Block(i);
				const auto flipped = ~block & mask;
				Block(i) = (block & ~mask) | flipped;
			}
		}

		constexpr bool IsAllOnes() const noexcept
		{
			if (Size() == 0)
			{
				return false;
			}

			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				if (const auto mask = MaskOfBlock(i); (mask & Block(i)) != mask)
				{
					return false;
				}
			}

			return true;
		}

		constexpr bool IsAllZeros() const noexcept
		{
			if (Size() == 0)
			{
				return false;
			}

			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				if ((MaskOfBlock(i) & Block(i)) != 0)
				{
					return false;
				}
			}

			return true;
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
			std::size_t count = 0;
			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				count += Detail::Bitset::PopCount(MaskOfBlock(i) & Block(i));
			}
			return count;
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
				if (const std::uint64_t block = Block(i) & MaskOfBlock(i); block != 0)
				{
					return i * Detail::Bitset::BitsPerBlock + Detail::Bitset::TrailingZeroCount(block) - HeadPadding();
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
				if (const std::uint64_t block = Block(i) & MaskOfBlock(i); block != 0)
				{
					return (BlockCount() - 1 - i) * Detail::Bitset::BitsPerBlock
					       + Detail::Bitset::LeadingZeroCount(block) - TailPadding();
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

			const std::size_t startBit = HeadPadding() + pos + 1;
			std::size_t blockIdx = startBit / Detail::Bitset::BitsPerBlock;
			const std::size_t bitInBlk = startBit % Detail::Bitset::BitsPerBlock;

			if (const std::uint64_t blk = Block(blockIdx) >> bitInBlk; blk != 0)
			{
				const std::size_t idx = startBit + Detail::Bitset::TrailingZeroCount(blk) - HeadPadding();
				// guards against over-counting caused by padding-bits leak into payload-bits segment
				return idx < Size() ? idx : Size();
			}

			for (++blockIdx; blockIdx < BlockCount(); ++blockIdx)
			{
				if (const std::uint64_t blk = Block(blockIdx); blk != 0)
				{
					const std::size_t idx = blockIdx * Detail::Bitset::BitsPerBlock
					                        + Detail::Bitset::TrailingZeroCount(blk) - HeadPadding();
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

			const std::size_t endBit = HeadPadding() + pos - 1;
			std::size_t blockIdx = endBit / Detail::Bitset::BitsPerBlock;
			const std::size_t bitInBlk = endBit % Detail::Bitset::BitsPerBlock;

			if (const std::uint64_t blk = Block(blockIdx) << (Detail::Bitset::BitsPerBlock - bitInBlk - 1); blk != 0)
			{
				const auto idx = blockIdx * Detail::Bitset::BitsPerBlock + bitInBlk
				                 - Detail::Bitset::LeadingZeroCount(blk) - HeadPadding();
				// guards against over-counting caused by padding-bits leak into payload-bits segment
				return idx < Size() ? idx : Size();
			}

			while (blockIdx-- > 0)
			{
				if (const std::uint64_t blk = Block(blockIdx); blk != 0)
				{
					const auto idx = blockIdx * Detail::Bitset::BitsPerBlock
					                 + (Detail::Bitset::BitsPerBlock - 1 - Detail::Bitset::LeadingZeroCount(blk))
					                 - HeadPadding();
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

		template <typename OtherDerived, typename Op>
		Derived& BitwiseCombinationOp(const BitsetBase<OtherDerived>& other, const Op op) SEC_NOEXCEPT
		{
			SEC_ASSERT(Size() == other.Size());

			const auto masks = MakeBlockMaskCaches();

			if (HeadPadding() == other.HeadPadding())
			{
				for (std::size_t i = 0; i < BlockCount(); ++i)
				{
					const auto mask = masks(i);
					const auto b0 = Block(i);
					const auto b = op(b0, other.Block(i));
					const auto n = (b & mask) | ~mask;
					Block(i) = (b0 & ~mask) | (n & mask);
				}

				return AsDerived();
			}

			const std::size_t thisHead = HeadPadding();
			const std::size_t otherHead = other.HeadPadding();

			// How many bits to shift other's raw data to align to *this.
			// Positive = shift left, negative = shift right.
			const std::ptrdiff_t shift = static_cast<std::ptrdiff_t>(thisHead) - static_cast<std::ptrdiff_t>(otherHead);
			SEC_ASSERT(shift != 0);

			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				const auto mask = masks(i);                   // valid-bit mask for *this's block i
				const auto b = Block(i);                      // current block in *this
				const auto o = other.ShiftedBlock(shift, i);  // other's bits aligned to *this
				const auto result = op(b, o);                 // AND/OR/XOR the valid bits

				// Preserve *this's padding bits (outside mask), write AND result inside mask.
				Block(i) = (b & ~mask) | (result & mask);
			}

			return AsDerived();
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

			const std::size_t blockShift = n / Detail::Bitset::BitsPerBlock;
			const std::size_t bitShift = n % Detail::Bitset::BitsPerBlock;
			const auto masks = MakeBlockMaskCaches();
			// const auto masks = [](const std::size_t) { return ~std::uint64_t{0}; };

			if (bitShift == 0)
			{
				{
					const std::size_t i = BlockCount() - 1;
					const auto mask = masks(i);
					const auto pad = Block(i) & ~mask;

					const std::size_t src = i - blockShift;
					const std::uint64_t val = Block(src);

					Block(i) = pad | (val & mask);
				}

				for (std::size_t i = BlockCount() - 1; BlockCount() >= 2 && i-- > blockShift;)
				{
					const std::size_t src = i - blockShift;
					Block(i) = Block(src);
				}
			}
			else
			{
				if (const std::size_t i = BlockCount() - 1; i > blockShift)
				{
					const auto mask = masks(i);
					const auto pad = Block(i) & ~mask;

					const std::size_t src = i - blockShift;
					std::uint64_t val = Block(src) << bitShift;
					const std::size_t carriedSrc = src - 1;
					val |= Block(carriedSrc) >> (Detail::Bitset::BitsPerBlock - bitShift);

					Block(i) = pad | (val & mask);
				}

				for (std::size_t i = BlockCount() - 2; BlockCount() >= 2 && i > blockShift; i--)
				{
					const std::size_t src = i - blockShift;
					const std::size_t carriedSrc = src - 1;
					Block(i) =
					        (Block(src) << bitShift) | (Block(carriedSrc) >> (Detail::Bitset::BitsPerBlock - bitShift));
				}

				{
					const std::size_t i = blockShift;
					const auto mask = masks(i);
					const auto pad = Block(i) & ~mask;

					const std::size_t src = i - blockShift;
					// no need for mask `Block(src)` since we'll clear the trailing later
					const std::uint64_t val = Block(src) << bitShift;

					Block(i) = pad | (val & mask);
				}
			}

			Trailing(n).ResetAll();

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

			const std::size_t blockShift = n / Detail::Bitset::BitsPerBlock;
			const std::size_t bitShift = n % Detail::Bitset::BitsPerBlock;
			const auto masks = MakeBlockMaskCaches();
			// const auto masks = [](const std::size_t) { return ~std::uint64_t{0}; };

			if (bitShift == 0)
			{
				{
					const auto mask = MaskOfBlock(0);
					const auto pad = Block(0) & ~mask;

					const std::size_t src = 0 + blockShift;
					const std::uint64_t val = Block(src);

					Block(0) = pad | (val & mask);
				}
				for (std::size_t i = 1; i + blockShift + 1 <= BlockCount(); i++)
				{
					const std::size_t src = i + blockShift;
					Block(i) = Block(src);
				}
			}
			else
			{
				if (constexpr std::size_t i = 0; i + blockShift + 1 < BlockCount())
				{
					const auto mask = masks(i);
					const auto pad = Block(i) & ~mask;

					const std::size_t src = i + blockShift;
					std::uint64_t val = Block(src) >> bitShift;
					const std::size_t carriedSrc = src + 1;
					val |= Block(carriedSrc) << (Detail::Bitset::BitsPerBlock - bitShift);

					Block(i) = pad | (val & mask);
				}

				for (std::size_t i = 1; i + blockShift + 1 < BlockCount(); ++i)
				{
					const std::size_t src = i + blockShift;
					const std::size_t carriedSrc = src + 1;
					Block(i) =
					        (Block(src) >> bitShift) | (Block(carriedSrc) << (Detail::Bitset::BitsPerBlock - bitShift));
				}

				{
					const std::size_t i = BlockCount() - blockShift - 1;
					const auto mask = MaskOfBlock(i);
					const auto pad = Block(i) & ~mask;

					const std::size_t src = i + blockShift;
					const std::uint64_t val = Block(src) >> bitShift;

					Block(i) = pad | (val & mask);
				}
			}

			Leading(n).ResetAll();

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

			if (HeadPadding() == other.HeadPadding())
			{
				SEC_ASSERT(BlockCount() == other.BlockCount());
				for (std::size_t i = 0; i < BlockCount(); ++i)
				{
					const auto mask = MaskOfBlock(i);
					if ((Block(i) & mask) != (other.Block(i) & mask))
					{
						return false;
					}
				}
				return true;
			}

			for (std::size_t i = 0; i < Detail::Bitset::BlocksFor(Size()); ++i)
			{
				const auto l = ShiftedBlock(-HeadPadding(), i);
				const auto r = other.ShiftedBlock(-other.HeadPadding(), i);

				if (i + 1 == Detail::Bitset::BlocksFor(Size()))
				{
					const auto mask = Detail::Bitset::LastBlockMask(Size());
					if ((l & mask) != (r & mask))
					{
						return false;
					}
				}
				else
				{
					if (l != r)
					{
						return false;
					}
				}
			}

			return true;
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
		// static constexpr std::size_t BitsPerBlock = Detail::Bitset::BitsPerBlock;
	};
}
