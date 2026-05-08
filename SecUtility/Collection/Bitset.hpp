// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Collection/Detail/Bitset.Forward.hpp>
#include <SecUtility/Collection/Detail/Bitset.Utility.hpp>
#include <SecUtility/Collection/Detail/BitsetBase.hpp>
#include <SecUtility/Collection/Detail/Bitset.Impl.hpp>

#if false

#include <SecUtility/Collection/SubscriptBasedIterator.hpp>
#include <SecUtility/IO/BitOrder.hpp>
#include <SecUtility/Macro/ForceInline.hpp>
#include <SecUtility/Math/Core.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
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

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winvalid-constexpr"
#endif

namespace SecUtility
{
	template <typename Derived>
	class BitsetBase;

	template <std::size_t N>
	class Bitset;

	template <typename Nested>
	class BitsetSegmentExpr;

	template <typename Nested>
	class BitsetNotExpr;

	template <typename Op, typename Lhs, typename Rhs>
	class BitsetBinaryExpr;

	class DynamicBitset;

	template <>
	struct Traits<DynamicBitset>;

	// ============================================================
	//  Internal helpers
	// ============================================================
	namespace Detail::Bitset
	{
		inline constexpr std::size_t BitsPerBlock = 64;

		constexpr std::size_t BlocksFor(const std::size_t bits) SEC_NOEXCEPT
		{
			return (bits + BitsPerBlock - 1) / BitsPerBlock;
		}

		// Mask of the active bits in the last block.  Returns ~0ull when bits % 64 == 0.
		constexpr std::uint64_t LastBlockMask(const std::size_t bits) SEC_NOEXCEPT
		{
			const std::size_t remainder = bits % BitsPerBlock;
			return remainder == 0 ? ~std::uint64_t{0} : (std::uint64_t{1} << remainder) - 1u;
		}

		constexpr std::size_t PopCount(std::uint64_t v) SEC_NOEXCEPT
		{
#if defined(__GNUC__) || defined(__clang__)
			// `long long` is at least 64 bits
			// ReSharper disable once CppRedundantCastExpression
			return static_cast<std::size_t>(__builtin_popcountll(static_cast<unsigned long long>(v)));
#elif defined(_MSC_VER)
			return static_cast<std::size_t>(__popcnt64(v));
#else
			v = v - ((v >> 1) & 0x5555555555555555ull);
			v = (v & 0x3333333333333333ull) + ((v >> 2) & 0x3333333333333333ull);
			v = (v + (v >> 4)) & 0x0F0F0F0F0F0F0F0Full;
			return static_cast<std::size_t>((v * 0x0101010101010101ull) >> 56);
#endif
		}

		template <typename T>
		std::size_t PopCount(T) = delete;  // to prevent conversion

		constexpr std::size_t TrailingZeroCount(std::uint64_t v) SEC_NOEXCEPT  // count trailing zeros, v != 0
		{
			if (v == 0)
			{
				return 64;
			}
			// SEC_ASSERT(v != 0);
#if defined(__GNUC__) || defined(__clang__)
			// `long long` is at least 64 bits
			// ReSharper disable once CppRedundantCastExpression
			return static_cast<std::size_t>(__builtin_ctzll(static_cast<unsigned long long>(v)));
#elif defined(_MSC_VER)
			return static_cast<std::size_t>(_tzcnt_u64(v));
#else
			std::size_t n = 0;
			while ((v & 1u) == 0)
			{
				v >>= 1;
				++n;
			}
			return n;
#endif
		}

		template <typename T>
		std::size_t TrailingZeroCount(T) = delete;  // to prevent conversion

		constexpr std::size_t LeadingZeroCount(std::uint64_t v) SEC_NOEXCEPT  // count leading zeros, v != 0
		{
			SEC_ASSERT(v != 0);
#if defined(__GNUC__) || defined(__clang__)
			// `long long` is at least 64 bits
			// ReSharper disable once CppRedundantCastExpression
			return static_cast<std::size_t>(__builtin_clzll(static_cast<unsigned long long>(v)));
#elif defined(_MSC_VER)
			return static_cast<std::size_t>(_lzcnt_u64(v));
#else
			std::size_t n = 0;
			while ((v & (std::uint64_t{1} << 63)) == 0)
			{
				v <<= 1;
				++n;
			}
			return n;
#endif
		}

		template <typename T>
		std::size_t LeadingZeroCount(T) = delete;  // to prevent conversion

		template <typename>
		struct is_fixed_size_bitset : std::false_type
		{
			/* NO CODE */
		};

		template <std::size_t N>
		struct is_fixed_size_bitset<SecUtility::Bitset<N>> : std::true_type
		{
			/* NO CODE */
		};
	}


	// ============================================================
	//  BitsetBase<Derived>
	// ============================================================
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

		friend BitsetSegmentExpr<Derived>;
		friend BitsetSegmentExpr<const Derived>;

		friend BitsetNotExpr<Derived>;
		friend BitsetNotExpr<const Derived>;

		template <typename, typename, typename>
		friend class BitsetBinaryExpr;

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
				const std::uint64_t hi = i > 0 ? Block(i - 1) >> (BitsPerBlock - lshift) : std::uint64_t{0};
				return hi | lo;
			}
			else
			{
				// Shift right: high bits come from Block(i), low bits come from Block(i+1)
				const auto rshift = static_cast<std::size_t>(-shift);
				const std::uint64_t hi = Block(i) >> rshift;
				const std::uint64_t lo =
				        i + 1 < BlockCount() ? Block(i + 1) << (BitsPerBlock - rshift) : std::uint64_t{0};
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
		constexpr BitsetSegmentExpr<Derived> Segment(const std::size_t start, const std::size_t count) noexcept
		{
			return {AsDerived(), start, count};
		}

		constexpr BitsetSegmentExpr<const Derived> Segment(const std::size_t start,
		                                                   const std::size_t count) const noexcept
		{
			return {AsDerived(), start, count};
		}

		constexpr BitsetSegmentExpr<Derived> Leading(const std::size_t count) noexcept
		{
			return {AsDerived(), Size() - count, count};
		}

		constexpr BitsetSegmentExpr<const Derived> Leading(const std::size_t count) const noexcept
		{
			return {AsDerived(), Size() - count, count};
		}

		constexpr BitsetSegmentExpr<Derived> Trailing(const std::size_t count) noexcept
		{
			return {AsDerived(), 0, count};
		}

		constexpr BitsetSegmentExpr<const Derived> Trailing(const std::size_t count) const noexcept
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

		BitsetNotExpr<const Derived> operator~() const noexcept
		{
			return BitsetNotExpr<const Derived>{AsDerived()};
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

			const std::size_t blockShift = n / BitsPerBlock;
			const std::size_t bitShift = n % BitsPerBlock;
			const auto masks = MakeBlockMaskCaches();

			for (std::size_t i = BlockCount(); i-- > 0;)
			{
				const auto mask = masks(i);
				const auto pad = Block(i) & ~mask;

				std::uint64_t val = 0;

				if (i >= blockShift)
				{
					const std::size_t src = i - blockShift;
					val = (Block(src) & masks(src)) << bitShift;

					if (bitShift > 0 && src > 0)
					{
						const std::size_t carriedSrc = src - 1;
						val |= (Block(carriedSrc) & masks(carriedSrc)) >> (BitsPerBlock - bitShift);
					}
				}

				Block(i) = pad | (val & mask);
			}

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

			const std::size_t blockShift = n / BitsPerBlock;
			const std::size_t bitShift = n % BitsPerBlock;
			const std::size_t count = BlockCount();
			const auto masks = MakeBlockMaskCaches();

			for (std::size_t i = 0; i < count; ++i)
			{
				const auto mask = masks(i);
				const auto pad = Block(i) & ~mask;

				std::uint64_t val = 0;

				if (const std::size_t src = i + blockShift; src < count)
				{
					val = (Block(src) & masks(src)) >> bitShift;

					if (bitShift > 0 && src + 1 < count)
					{
						const std::size_t carriedSrc = src + 1;
						val |= (Block(carriedSrc) & masks(carriedSrc)) << (BitsPerBlock - bitShift);
					}
				}

				Block(i) = pad | (val & mask);
			}

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
		static constexpr std::size_t BitsPerBlock = Detail::Bitset::BitsPerBlock;
	};


	// ============================================================
	//  Bitset<N>  --  compile-time fixed size
	// ============================================================
	template <std::size_t N>
	class Bitset : public BitsetBase<Bitset<N>>
	{
		using Base = BitsetBase<Bitset>;
		friend Base;

		static constexpr std::size_t kN = N;
		static constexpr std::size_t kBlockCount =
		        (N + Detail::Bitset::BitsPerBlock - 1) / Detail::Bitset::BitsPerBlock;

		// ReSharper disable once CppMemberFunctionMayBeStatic
		/* CRTP OVERRIDE */ constexpr std::size_t HeadPadding() const noexcept  // corresponds to trailing
		{
			return 0;
		}

		/* CRTP OVERRIDE */ constexpr std::uint64_t& Block(const std::size_t index) noexcept
		{
			return m_Data[index];
		}

		/* CRTP OVERRIDE */ constexpr std::uint64_t Block(const std::size_t index) const noexcept
		{
			return m_Data[index];
		}

	public:
		// ----------------------------------------------------------
		//  Special functions
		// ----------------------------------------------------------
		constexpr Bitset(const Bitset& other) noexcept = default;
		constexpr Bitset(Bitset&& other) noexcept = default;
		constexpr Bitset& operator=(const Bitset& other) noexcept = default;
		constexpr Bitset& operator=(Bitset&& other) noexcept = default;
		~Bitset() noexcept = default;

		using Base::operator=;

		explicit Bitset(const bool value = false) noexcept : m_Data{}
		{
			if (value)
			{
				for (auto& block : m_Data)
				{
					block = ~std::uint64_t{0};
				}

				Base::RestPaddingBitsOfLastBlock();
			}
		}

		template <typename OtherDerived,
		          typename...,
		          typename = std::enable_if_t<!std::is_same_v<std::decay_t<OtherDerived>, Bitset>, void>>
		/* IMPLICIT */ constexpr Bitset(const BitsetBase<OtherDerived>& other) noexcept(
		        noexcept(Base::operator=(other)))
		    : m_Data{}
		{
			Base::operator=(other);
		}

		/* IMPLICIT */ constexpr Bitset(const BitsetBase<Bitset>& other) noexcept : Bitset(other.AsDerived())
		{
			/* NO CODE */
		}

		/* IMPLICIT */ constexpr Bitset(BitsetBase<Bitset>&& other) noexcept : Bitset(std::move(other.AsDerived()))
		{
			/* NO CODE */
		}

		// ----------------------------------------------------------
		//  Size / capacity
		// ----------------------------------------------------------
		// ReSharper disable once CppMemberFunctionMayBeStatic
		/* CRTP OVERRIDE */ constexpr std::size_t Size() const noexcept
		{
			return kN;
		}

	private:
		std::uint64_t m_Data[kBlockCount == 0 ? 1 : kBlockCount];
	};

	template <std::size_t N>
	struct Traits<Bitset<N>>
	{
		static constexpr bool IsNestedByRef = true;
		using EvaluatedType = Bitset<N>;
	};


	// ============================================================
	//  DynamicBitset  --  runtime-determined size
	// ============================================================
	class DynamicBitset : public BitsetBase<DynamicBitset>
	{
		using Base = BitsetBase;
		friend Base;

		// ReSharper disable once CppMemberFunctionMayBeStatic
		/* CRTP OVERRIDE */ constexpr std::size_t HeadPadding()
		        const noexcept  // corresponds to trailing NOLINT(*-convert-member-functions-to-static)
		{
			return 0;
		}

		/* CRTP OVERRIDE */ SEC_FORCE_INLINE constexpr std::uint64_t& Block(const std::size_t index) noexcept
		{
			return m_Data[index];
		}

		/* CRTP OVERRIDE */ SEC_FORCE_INLINE constexpr std::uint64_t Block(const std::size_t index) const noexcept
		{
			return m_Data[index];
		}


	public:
		// ----------------------------------------------------------
		//  Special functions
		// ----------------------------------------------------------
		DynamicBitset() : m_Size(0)
		{
			/* NO CODE */
		}

		DynamicBitset(const DynamicBitset& other) = default;
		DynamicBitset(DynamicBitset&& other) noexcept = default;
		DynamicBitset& operator=(const DynamicBitset& other) = default;
		DynamicBitset& operator=(DynamicBitset&& other) noexcept = default;
		~DynamicBitset() noexcept = default;

		using Base::operator=;

		template <typename OtherDerived,
		          typename...,
		          typename = std::enable_if_t<!std::is_same_v<std::decay_t<OtherDerived>, DynamicBitset>, void>>
		/* IMPLICIT */ DynamicBitset(const BitsetBase<OtherDerived>& other)
		    : m_Size(other.Size()), m_Data(Detail::Bitset::BlocksFor(m_Size))
		{
			Base::operator=(other);
		}

		/* IMPLICIT */ DynamicBitset(const BitsetBase& other) noexcept
		    : m_Size(other.AsDerived().m_Size), m_Data(other.AsDerived().m_Data)
		{
			/* NO CODE */
		}

		/* IMPLICIT */ DynamicBitset(BitsetBase&& other) noexcept
		    : m_Size(other.AsDerived().m_Size), m_Data(std::move(other.AsDerived().m_Data))
		{
			/* NO CODE */
		}

		template <typename Integer,
		          typename...,
		          typename = std::enable_if_t<std::is_integral_v<std::decay_t<Integer>>, void>>
		explicit DynamicBitset(const Integer size, const bool value = false)
		    : m_Size(
		              [size]
		              {
			              SEC_ASSERT(size >= 0);
			              return size;
		              }()),
		      m_Data(Detail::Bitset::BlocksFor(size), value ? ~std::uint64_t{0} : std::uint64_t{0})
		{
			if (value)
			{
				RestPaddingBitsOfLastBlock();
			}
		}

		// ----------------------------------------------------------
		//  Size / capacity
		// ----------------------------------------------------------
		/* CRTP OVERRIDE */ constexpr std::size_t Size() const SEC_NOEXCEPT
		{
			return m_Size;
		}

#if false
		void Reserve(const std::size_t size)
		{
			m_Data.reserve(Detail::Bitset::BlocksFor(size));
		}

		void Resize(const std::size_t newSize, const bool value = false)
		{
			const std::size_t newBc = Detail::Bitset::BlocksFor(newSize);
			const std::uint64_t fill = value ? ~std::uint64_t{0} : std::uint64_t{0};
			m_Data.resize(newBc, fill);
			m_Size = newSize;
			SanitizeLastBlock();  // clears padding in new last block when growing with value=true,
			                      // and trims old data when shrinking
		}

		void Clear() SEC_NOEXCEPT
		{
			m_Data.clear();
			m_Size = 0;
		}

		// ----------------------------------------------------------
		//  Advanced / utilities
		// ----------------------------------------------------------
		void PushBack(const bool value)
		{
			const std::size_t newIdx = m_Size;
			const std::size_t blkIdx = newIdx / Detail::Bitset::BitsPerBlock;
			if (blkIdx >= m_Data.size())
			{
				m_Data.push_back(0);
			}
			++m_Size;
			if (value)
			{
				m_Data[blkIdx] |= std::uint64_t{1} << (newIdx % Detail::Bitset::BitsPerBlock);
			}
		}

		template <typename OtherDerived>
		DynamicBitset& Append(const BitsetBase<OtherDerived>& other)
		{
			const std::size_t mySize = m_Size;
			const std::size_t otherSize = other.Size();
			if (otherSize == 0)
			{
				return *this;
			}

			Resize(mySize + otherSize, false);

			for (std::size_t i = 0; i < otherSize; ++i)
			{
				if (other[i])
				{
					const std::size_t dstIdx = mySize + i;
					m_Data[dstIdx / Detail::Bitset::BitsPerBlock] |= std::uint64_t{1}
					                                                 << (dstIdx % Detail::Bitset::BitsPerBlock);
				}
			}

			return *this;
		}
#endif

	private:
		std::size_t m_Size;
		std::vector<std::uint64_t> m_Data;
	};

	template <>
	struct Traits<DynamicBitset>
	{
		static constexpr bool IsNestedByRef = true;
		using EvaluatedType = DynamicBitset;
	};


	// ============================================================
	//  BitsetSegmentExpr  --  segment of existing bitset
	// ============================================================
	template <typename Nested>
	class BitsetSegmentExpr : public BitsetBase<BitsetSegmentExpr<Nested>>
	{
		using Base = BitsetBase<BitsetSegmentExpr>;
		friend Base;

		friend BitsetBase<BitsetSegmentExpr<std::remove_const_t<Nested>>>;
		friend BitsetBase<std::remove_const_t<Nested>>;
		friend BitsetBase<Nested>;

	private:
		friend Nested;
		friend BitsetBase<Nested>;

		using BaseOfNested = std::conditional_t<std::is_const_v<Nested>,
		                                        const BitsetBase<std::remove_const_t<Nested>>,
		                                        BitsetBase<Nested>>;

		constexpr BitsetSegmentExpr(Nested& nested, const std::size_t start, const std::size_t size) SEC_NOEXCEPT
		    : m_Nested(nested),
		      m_HeadPadding((start + static_cast<BaseOfNested&>(nested).HeadPadding()) % Detail::Bitset::BitsPerBlock),
		      m_BlockIndexOffset((start + static_cast<BaseOfNested&>(nested).HeadPadding())
		                         / Detail::Bitset::BitsPerBlock),
		      m_Size(size)
		{
			SEC_ASSERT(start + size <= nested.Size());
		}

		template <typename..., bool IsConst = std::is_const_v<Nested>>
		/* CRTP VIRTUAL */ constexpr std::enable_if_t<!IsConst, std::uint64_t&> Block(const std::size_t index) noexcept
		{
			return static_cast<BaseOfNested&>(m_Nested).Block(index + m_BlockIndexOffset);
		}

		/* CRTP VIRTUAL */ constexpr std::uint64_t Block(const std::size_t index) const noexcept
		{
			return static_cast<const BaseOfNested&>(m_Nested).Block(index + m_BlockIndexOffset);
		}

		/* CRTP VIRTUAL */ constexpr std::size_t HeadPadding() const noexcept  // corresponds to trailing
		{
			return m_HeadPadding;
		}


	public:
		/* CRTP VIRTUAL */ constexpr std::size_t Size() const noexcept
		{
			return m_Size;
		}

		using Base::operator=;

		BitsetSegmentExpr() = delete;
		constexpr BitsetSegmentExpr(const BitsetSegmentExpr&) noexcept = default;
		constexpr BitsetSegmentExpr(BitsetSegmentExpr&&) noexcept = default;
		constexpr BitsetSegmentExpr& operator=(BitsetSegmentExpr&&) = delete;
		~BitsetSegmentExpr() noexcept = default;

		constexpr BitsetSegmentExpr& operator=(const BitsetSegmentExpr& other) noexcept(
		        noexcept(Base::operator=(other)))
		{
			Base::operator=(other);
			return *this;
		}


	private:
		std::conditional_t<Traits<std::decay_t<Nested>>::IsNestedByRef, Nested&, Nested> m_Nested;
		std::size_t m_HeadPadding;
		std::size_t m_BlockIndexOffset;
		std::size_t m_Size;
	};

	template <typename Nested>
	struct Traits<BitsetSegmentExpr<Nested>>
	{
		static constexpr bool IsNestedByRef = false;
		using EvaluatedType = DynamicBitset;
	};

	// ============================================================
	//  BitsetNotExpr  --  bitwise not of existing bitset
	// ============================================================
	template <typename Nested>
	class BitsetNotExpr : public BitsetBase<BitsetNotExpr<Nested>>
	{
		using Base = BitsetBase<BitsetNotExpr>;
		friend Base;

		friend BitsetBase<BitsetNotExpr<std::remove_const_t<Nested>>>;
		friend BitsetBase<std::remove_const_t<Nested>>;
		friend BitsetBase<Nested>;

	private:
		friend Nested;
		friend BitsetBase<Nested>;

		using BaseOfNested = std::conditional_t<std::is_const_v<Nested>,
		                                        const BitsetBase<std::remove_const_t<Nested>>,
		                                        BitsetBase<Nested>>;

		explicit constexpr BitsetNotExpr(const Nested& nested) noexcept : m_Nested(nested)
		{
			/* NO CODE */
		}

		// NOLINTNEXTLINE(*-use-equals-delete)
		/* CRTP VIRTUAL */ constexpr std::uint64_t& Block(std::size_t index) noexcept = delete;

		/* CRTP VIRTUAL */ constexpr std::uint64_t Block(const std::size_t index) const noexcept
		{
			return ~static_cast<BaseOfNested&>(m_Nested).Block(index);
		}

		/* CRTP VIRTUAL */ constexpr std::size_t HeadPadding() const noexcept  // corresponds to trailing
		{
			return static_cast<BaseOfNested&>(m_Nested).HeadPadding();
		}


	public:
		/* CRTP VIRTUAL */ constexpr std::size_t Size() const noexcept
		{
			return m_Nested.Size();
		}

		BitsetNotExpr() = delete;
		constexpr BitsetNotExpr(const BitsetNotExpr&) noexcept = default;
		constexpr BitsetNotExpr(BitsetNotExpr&&) noexcept = default;
		BitsetNotExpr& operator=(const BitsetNotExpr&) = delete;
		BitsetNotExpr& operator=(BitsetNotExpr&&) = delete;
		~BitsetNotExpr() noexcept = default;

	private:
		std::conditional_t<Traits<std::decay_t<Nested>>::IsNestedByRef, Nested&, Nested> m_Nested;
	};

	template <typename Nested>
	struct Traits<BitsetNotExpr<Nested>>
	{
		static constexpr bool IsNestedByRef = false;
		using EvaluatedType = typename Traits<std::remove_const_t<Nested>>::EvaluatedType;
	};

	// ============================================================
	//  BitsetAndExpr  --  bitwise and
	// ============================================================
	template <typename Op, typename Lhs, typename Rhs>
	class BitsetBinaryExpr : public BitsetBase<BitsetBinaryExpr<Op, Lhs, Rhs>>
	{
		using Base = BitsetBase<BitsetBinaryExpr>;
		friend Base;

		friend BitsetBase<BitsetBinaryExpr<Op, std::remove_const_t<Lhs>, Rhs>>;
		friend BitsetBase<BitsetBinaryExpr<Op, Lhs, std::remove_const_t<Rhs>>>;
		friend BitsetBase<BitsetBinaryExpr<Op, std::remove_const_t<Lhs>, std::remove_const_t<Rhs>>>;

		friend Lhs;
		friend BitsetBase<Lhs>;

		using BaseOfLhs =
		        std::conditional_t<std::is_const_v<Lhs>, const BitsetBase<std::remove_const_t<Lhs>>, BitsetBase<Lhs>>;
		using BaseOfRhs =
		        std::conditional_t<std::is_const_v<Rhs>, const BitsetBase<std::remove_const_t<Rhs>>, BitsetBase<Rhs>>;

	public:
		explicit constexpr BitsetBinaryExpr(const Lhs& lhs,
		                                    const Rhs& rhs,
		                                    const std::size_t headPadding = 0) SEC_NOEXCEPT
		    : m_HeadPadding(headPadding),
		      m_Lhs(lhs.AsDerived()),
		      m_Rhs(rhs.AsDerived())
		{
			SEC_ASSERT(lhs.Size() == rhs.Size());
		}

	private:
		// NOLINTNEXTLINE(*-use-equals-delete)
		/* CRTP VIRTUAL */ constexpr std::uint64_t& Block(std::size_t index) noexcept = delete;

		/* CRTP VIRTUAL */ constexpr std::uint64_t Block(const std::size_t index) const noexcept
		{
			const auto& l = static_cast<const BaseOfLhs&>(m_Lhs);
			const auto& r = static_cast<const BaseOfRhs&>(m_Rhs);
			return Op{}(l.ShiftedBlock(m_HeadPadding - l.HeadPadding(), index),
			            r.ShiftedBlock(m_HeadPadding - r.HeadPadding(), index));
		}

		/* CRTP VIRTUAL */ constexpr std::size_t HeadPadding() const noexcept  // corresponds to trailing
		{
			return m_HeadPadding;
		}

	public:
		/* CRTP VIRTUAL */ constexpr std::size_t Size() const noexcept
		{
			return m_Lhs.Size();
		}

		BitsetBinaryExpr() = delete;
		constexpr BitsetBinaryExpr(const BitsetBinaryExpr&) noexcept = default;
		constexpr BitsetBinaryExpr(BitsetBinaryExpr&&) noexcept = default;
		BitsetBinaryExpr& operator=(const BitsetBinaryExpr&) = delete;
		BitsetBinaryExpr& operator=(BitsetBinaryExpr&&) = delete;
		~BitsetBinaryExpr() noexcept = default;

	private:
		std::size_t m_HeadPadding{};
		std::conditional_t<Traits<std::decay_t<Lhs>>::IsNestedByRef, const Lhs&, Lhs> m_Lhs;
		std::conditional_t<Traits<std::decay_t<Rhs>>::IsNestedByRef, const Rhs&, Rhs> m_Rhs;
	};

	template <typename Op, typename Lhs, typename Rhs>
	struct Traits<BitsetBinaryExpr<Op, Lhs, Rhs>>
	{
		static constexpr bool IsNestedByRef = false;
		using EvaluatedType = std::conditional_t<Detail::Bitset::is_fixed_size_bitset<std::decay_t<Rhs>>::value,
		                                         typename Traits<std::remove_const_t<Rhs>>::EvaluatedType,
		                                         typename Traits<std::remove_const_t<Lhs>>::EvaluatedType>;
	};

	// ============================================================
	//  Binary bitwise operators
	// ============================================================
	template <typename Derived>
	auto operator<<(const BitsetBase<Derived>& bs, const std::size_t size)
	{
		typename Traits<Derived>::EvaluatedType r = bs;
		r <<= size;
		return r;
	}

	template <typename Derived>
	auto operator>>(const BitsetBase<Derived>& bs, const std::size_t size)
	{
		typename Traits<Derived>::EvaluatedType r = bs;
		r >>= size;
		return r;
	}

#define SEC_DEFINE_BITSET_BITWISE_BINARY_OP(OP)                                                                        \
	template <typename Lhs, typename Rhs>                                                                              \
	constexpr auto operator OP(const BitsetBase<Lhs>& lhs, const BitsetBase<Rhs>& rhs) SEC_NOEXCEPT                    \
	{                                                                                                                  \
		if constexpr (Detail::Bitset::is_fixed_size_bitset<typename Traits<Rhs>::EvaluatedType>::value)                \
		{                                                                                                              \
			typename Traits<Rhs>::EvaluatedType r{rhs};                                                                \
			r OP## = lhs;                                                                                              \
			return r;                                                                                                  \
		}                                                                                                              \
		else                                                                                                           \
		{                                                                                                              \
			typename Traits<Lhs>::EvaluatedType r{lhs};                                                                \
			r OP## = rhs;                                                                                              \
			return r;                                                                                                  \
		}                                                                                                              \
	}

	SEC_DEFINE_BITSET_BITWISE_BINARY_OP(&)
	SEC_DEFINE_BITSET_BITWISE_BINARY_OP(|)
	SEC_DEFINE_BITSET_BITWISE_BINARY_OP(^)

#undef SEC_DEFINE_BITSET_BITWISE_BINARY_OP
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#endif