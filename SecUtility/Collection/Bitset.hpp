// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <SecUtility/Collection/SubscriptBasedIterator.hpp>
#include <SecUtility/IO/BitOrder.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

	template <std::size_t N>  // N to avoid shadowing the Size() method
	class Bitset;

	template <typename Nested>
	class BitsetSegmentExpr;

	class DynamicBitset;

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

		constexpr std::size_t CountTrailingZeros(std::uint64_t v) SEC_NOEXCEPT  // count trailing zeros, v != 0
		{
			SEC_ASSERT(v != 0);
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
		std::size_t CountTrailingZeros(T) = delete;  // to prevent conversion

		constexpr std::size_t CountLeadingZeros(std::uint64_t v) SEC_NOEXCEPT  // count leading zeros, v != 0
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
		std::size_t CountLeadingZeros(T) = delete;  // to prevent coversion

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

		constexpr const Derived& AsDerived() const noexcept
		{
			return static_cast<const Derived&>(*this);
		}

		constexpr Derived& AsDerived() noexcept
		{
			return static_cast<Derived&>(*this);
		}

		constexpr BitsetBase() noexcept = default;
		constexpr BitsetBase(const BitsetBase&) noexcept = default;
		constexpr BitsetBase(BitsetBase&&) noexcept = default;
		constexpr BitsetBase& operator=(const BitsetBase&) noexcept = default;
		constexpr BitsetBase& operator=(BitsetBase&&) noexcept = default;
		~BitsetBase() noexcept = default;

		// Forwarding helpers
		/* CRTP VIRTUAL */ constexpr std::size_t BlockCount() const noexcept
		{
			return AsDerived().BlockCount();
		}

		/* CRTP VIRTUAL */ constexpr std::uint64_t& Block(const std::size_t index) noexcept
		{
			return AsDerived().Block(index);
		}

		/* CRTP VIRTUAL */ constexpr std::uint64_t Block(const std::size_t index) const noexcept
		{
			return AsDerived().Block(index);
		}

		/* CRTP VIRTUAL */ constexpr std::size_t HeadPadding() const SEC_NOEXCEPT  // corresponds to trailing
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

		constexpr void RestPaddingBitsOfZerothBlock() noexcept
		{
			if (const std::size_t size = Size(); size != 0)
			{
				Block(BlockCount() - 1) &= ~Detail::Bitset::LastBlockMask(HeadPadding());
			}
		}

		constexpr std::uint64_t MaskOfBlock(const std::size_t index) const SEC_NOEXCEPT
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

		constexpr void RestPaddingBitsOfLastBlock() noexcept
		{
			if (const std::size_t paddedSize = Size() + HeadPadding(); paddedSize != 0)
			{
				Block(BlockCount() - 1) &= Detail::Bitset::LastBlockMask(paddedSize);
			}
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
				return static_cast<bool>(r_Derived.Block(m_PaddedIndex / Detail::Bitset::BitsPerBlock) & m_Mask);
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
		// ----------------------------------------------------------
		//  Size / capacity
		// ----------------------------------------------------------
		constexpr std::size_t Size() const noexcept
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

		constexpr auto rbegin() noexcept
		{
			return std::reverse_iterator{end()};
		}

		constexpr auto rbegin() const noexcept
		{
			return std::reverse_iterator{end()};
		}

		constexpr auto rcbegin() const noexcept
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

		constexpr auto rcend() const noexcept
		{
			return std::reverse_iterator{cbegin()};
		}


	public:
		constexpr BitsetSegmentExpr<Derived> Segment(const std::size_t start, const std::size_t count) noexcept
		{
			return {AsDerived(), start, count};
		}

		constexpr BitsetSegmentExpr<const Derived> Segment(const std::size_t start, const std::size_t count) const noexcept
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
			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				if (value)
				{
					Block(i) |= MaskOfBlock(i);
				}
				else
				{
					Block(i) &= ~MaskOfBlock(i);
				}
			}
		}

		constexpr void ResetAll() noexcept
		{
			SetAll(false);
		}

		constexpr void FlipAll() noexcept
		{
			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				const auto mask = MaskOfBlock(i);
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

#if false
		std::size_t TrailingZeroCount() const noexcept;

		std::size_t TrailingOneCount() const noexcept;

		std::size_t LeadingZeroCount() const noexcept;

		std::size_t LeadingOneCount() const noexcept;

		// ----------------------------------------------------------
		//  Index-of queries  (return Size() when not found)
		// ----------------------------------------------------------
		std::size_t IndexOfFirstOne() const noexcept
		{
			return TrailingZeroCount();
		}

		std::size_t IndexOfFirstZero() const noexcept
		{
			return TrailingOneCount();
		}

		std::size_t IndexOfLastOne() const noexcept
		{
			const std::size_t lz = LeadingZeroCount();
			return lz == Size() ? Size() : Size() - 1 - lz;
		}

		std::size_t IndexOfLastZero() const noexcept
		{
			const std::size_t lo = LeadingOneCount();
			return lo == Size() ? Size() : Size() - 1 - lo;
		}

		std::size_t IndexOfNextOne(const std::size_t pos) const noexcept;

		std::size_t IndexOfNextZero(const std::size_t pos) const noexcept;

		std::size_t IndexOfPreviousOne(const std::size_t pos) const noexcept;

		std::size_t IndexOfPreviousZero(const std::size_t pos) const noexcept;

		// ----------------------------------------------------------
		//  Bitwise operations
		// ----------------------------------------------------------
		template <typename OtherDerived>
		Derived& operator&=(const BitsetBase<OtherDerived>& other) SEC_NOEXCEPT;

		template <typename OtherDerived>
		Derived& operator|=(const BitsetBase<OtherDerived>& other) SEC_NOEXCEPT;

		template <typename OtherDerived>
		Derived& operator^=(const BitsetBase<OtherDerived>& other) SEC_NOEXCEPT;

		CompExpr<const Derived> operator~() const noexcept
		{
			return CompExpr<const Derived>{AsDerived()};
		}

		// ----------------------------------------------------------
		//  Shift operations  (bit 0 == LSB of Data()[0])
		//  <<  shifts bits toward higher indices ("left" in math notation)
		//  >>  shifts bits toward lower  indices
		// ----------------------------------------------------------
		Derived& operator<<=(const std::size_t shift);

		Derived& operator>>=(const std::size_t shift);

		LeftShiftExpr<const Derived> operator<<(std::size_t shift) const;

		RightShiftExpr<const Derived> operator>>(std::size_t shift) const;
#endif

#if false
		// ----------------------------------------------------------
		//  Comparison
		// ----------------------------------------------------------
		template <typename OtherDerived>
		bool operator==(const BitsetBase<OtherDerived>& other) const noexcept;

		template <typename OtherDerived>
		bool operator!=(const BitsetBase<OtherDerived>& other) const noexcept
		{
			return !(*this == other.AsDerived());
		}
#endif

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


	protected:
		template <typename Fn>
		void ApplyToAll(Fn fn) SEC_NOEXCEPT
		{
			if (Size() == 0)
			{
				return;
			}

			std::size_t bitIdx = HeadPadding(), remain = Size();
			while (remain > 0)
			{
				const std::size_t blkIdx = bitIdx / Detail::Bitset::BitsPerBlock;
				const std::size_t bitInBlk = bitIdx % Detail::Bitset::BitsPerBlock;
				const std::size_t bitsNow = std::min(Detail::Bitset::BitsPerBlock - bitInBlk, remain);
				const std::uint64_t mask = bitsNow == Detail::Bitset::BitsPerBlock
				                                   ? ~std::uint64_t{0}
				                                   : ((std::uint64_t{1} << bitsNow) - 1u) << bitInBlk;
				fn(Block(bitIdx), mask);
				bitIdx += bitsNow;
				remain -= bitsNow;
			}
		}

		// const overload: casts away const so we can reuse the mutable version,
		// but the lambda must only read.
		template <typename Fn>
		constexpr void ApplyToAll(Fn fn) const noexcept(noexcept(fn(std::uint64_t{}, std::uint64_t{})))
		{
			const_cast<BitsetBase*>(this)->ApplyToAll([&fn](const std::uint64_t block, const std::uint64_t mask)
			                                          { fn(block, mask); });
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


		/* CRTP OVERRIDE */ constexpr std::size_t BlockCount() const noexcept
		{
			return kBlockCount;
		}

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
		//  Ctors
		// ----------------------------------------------------------
		constexpr Bitset(const Bitset& other) noexcept = default;
		constexpr Bitset(Bitset&& other) noexcept = default;
		constexpr Bitset& operator=(const Bitset& other) noexcept = default;
		constexpr Bitset& operator=(Bitset&& other) noexcept = default;
		~Bitset() noexcept = default;

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


#if false
		template <typename OtherDerived>
		explicit Bitset(const BitsetBase<OtherDerived>& other) : m_Data{}
		{
			SEC_ASSERT(other.Size() == N);
			for (std::size_t i = 0; i < kBlockCount; ++i)
			{
				m_Data[i] = other.Data()[i];
			}
			SanitizeLastBlock();
		}

		template <typename OtherDerived>
		constexpr Bitset& operator=(const BitsetBase<OtherDerived>& other)
		{
			SEC_ASSERT(other.Size() == N);
			for (std::size_t i = 0; i < kBlockCount; ++i)
			{
				m_Data[i] = other.Data()[i];
			}
			SanitizeLastBlock();
			return *this;
		}

		template <typename OtherDerived>
		constexpr Bitset& operator=(BitsetBase<OtherDerived>&& other)
		{
			return *this = static_cast<const BitsetBase<OtherDerived>&>(other);
		}
#endif

		// ----------------------------------------------------------
		//  Size / capacity
		// ----------------------------------------------------------
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
	};


	// ============================================================
	//  DynamicBitset  --  runtime-determined size
	// ============================================================
	class DynamicBitset : public BitsetBase<DynamicBitset>
	{
		using Base = BitsetBase;
		friend Base;

		/* CRTP OVERRIDE */ constexpr std::size_t HeadPadding() const noexcept  // corresponds to trailing
		{
			return 0;
		}

		/* CRTP OVERRIDE */ constexpr std::size_t BlockCount() const SEC_NOEXCEPT
		{
			return m_Data.size();
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
		DynamicBitset() : m_Size(0)
		{
			/* NO CODE */
		}

		DynamicBitset(const DynamicBitset& other) = default;
		DynamicBitset(DynamicBitset&& other) noexcept = default;
		DynamicBitset& operator=(const DynamicBitset& other) = default;
		DynamicBitset& operator=(DynamicBitset&& other) noexcept = default;
		~DynamicBitset() noexcept = default;

#if false
		template <typename OtherDerived>
		explicit DynamicBitset(const BitsetBase<OtherDerived>& other)
		    : m_Size(other.Size()), m_Data(other.Data(), other.Data() + other.BlockCount())
		{
			SanitizeLastBlock();
		}

		template <typename OtherDerived>
		constexpr DynamicBitset& operator=(const BitsetBase<OtherDerived>& other)
		{
			m_Size = other.Size();
			m_Data.assign(other.Data(), other.Data() + other.BlockCount());
			SanitizeLastBlock();
			return *this;
		}
#endif

		template <typename Integer>
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

		/* CRTP VIRTUAL */ constexpr std::size_t BlockCount() const noexcept
		{
			return (m_HeadPadding + m_Size + Detail::Bitset::BitsPerBlock - 1) / Detail::Bitset::BitsPerBlock;
		}

		template <typename..., bool IsConst = std::is_const_v<Nested>>
		/* CRTP VIRTUAL */ constexpr std::enable_if_t<!IsConst, std::uint64_t&> Block(const std::size_t index) noexcept
		{
			return static_cast<BaseOfNested&>(m_Nested).Block(index + m_BlockIndexOffset);
		}

		/* CRTP VIRTUAL */ constexpr std::uint64_t Block(const std::size_t index) const noexcept
		{
			return static_cast<BaseOfNested&>(m_Nested).Block(index + m_BlockIndexOffset);
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
	};
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
