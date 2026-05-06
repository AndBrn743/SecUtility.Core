// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <SecUtility/Collection/SubscriptBasedIterator.hpp>

#if !defined(SEC_ASSERT)
#define SEC_ASSERT(expr) assert(expr)
#endif
#if !defined(SEC_NOEXCEPT)
#define SEC_NOEXCEPT noexcept
#endif

namespace SecUtility
{
	template <typename Derived>
	class BitsetBase;

	template <std::size_t N>  // N to avoid shadowing the Size() method
	class Bitset;

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

		/* CRTP VIRTUAL */ constexpr std::size_t HeadPadding() const noexcept  // corresponds to trailing
		{
			return AsDerived().HeadPadding();
		}

		constexpr std::size_t TailPadding() const noexcept  // corresponds to leading
		{
			return BlockCount() * Detail::Bitset::BitsPerBlock - (Size() + HeadPadding());
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
		bool operator[](const std::size_t index) const SEC_NOEXCEPT
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

		Derived& Reset(const std::size_t index) SEC_NOEXCEPT
		{
			SEC_ASSERT(index < Size());
			(*this)[index] = false;
			return AsDerived();
		}

		Derived& Flip(const std::size_t index) SEC_NOEXCEPT
		{
			SEC_ASSERT(index < Size());
			(*this)[index].Flip();
			return AsDerived();
		}

		// ----------------------------------------------------------
		//  Iterator
		// ----------------------------------------------------------
		auto begin() noexcept
		{
			return SubscriptBasedIterator<Derived>{AsDerived(), 0};
		}

		auto begin() const noexcept
		{
			return SubscriptBasedIterator<const Derived>{AsDerived(), 0};
		}

		auto cbegin() const noexcept  // NOLINT
		{
			return SubscriptBasedIterator<const Derived>{AsDerived(), 0};
		}

		auto end() noexcept
		{
			return SubscriptBasedIterator<Derived>{AsDerived(), Size()};
		}

		auto end() const noexcept
		{
			return SubscriptBasedIterator<const Derived>{AsDerived(), Size()};
		}

		auto cend() const noexcept  // NOLINT
		{
			return SubscriptBasedIterator<const Derived>{AsDerived(), Size()};
		}

		auto rbegin() noexcept
		{
			return std::reverse_iterator{SubscriptBasedIterator<Derived>{AsDerived(), 0}};
		}

		auto rbegin() const noexcept
		{
			return std::reverse_iterator{SubscriptBasedIterator<const Derived>{AsDerived(), 0}};
		}


#if false
		// ----------------------------------------------------------
		//  Whole-set operations
		// ----------------------------------------------------------
		void SetAll(const bool value = true) noexcept;

		void ResetAll() noexcept
		{
			SetAll(false);
		}

		void FlipAll() noexcept;

		bool IsAllOnes() const noexcept;

		bool IsAllZeros() const noexcept;

		bool HasOnes() const noexcept
		{
			return !IsEmpty() && !IsAllZeros();
		}

		bool HasZeros() const noexcept
		{
			return !IsEmpty() && !IsAllOnes();
		}

	public:
		SegmentExpr<Derived> Segment(const std::size_t start, const std::size_t count) noexcept
		{
			return {AsDerived(), start, count};
		}

		SegmentExpr<const Derived> Segment(const std::size_t start, const std::size_t count) const noexcept
		{
			return {AsDerived(), start, count};
		}

		SegmentExpr<Derived> Leading(const std::size_t count) noexcept
		{
			return {AsDerived(), 0, count};
		}

		SegmentExpr<const Derived> Leading(const std::size_t start, const std::size_t count) const noexcept
		{
			return {AsDerived(), 0, count};
		}

		SegmentExpr<Derived> Trailing(const std::size_t count) noexcept
		{
			return {AsDerived(), Size() - count, count};
		}

		SegmentExpr<const Derived> Trailing(const std::size_t start, const std::size_t count) const noexcept
		{
			return {AsDerived(), Size() - count, count};
		}

		// ----------------------------------------------------------
		//  Counting
		// ----------------------------------------------------------
		std::size_t CountOnes() const noexcept;

		std::size_t CountZeros() const noexcept
		{
			return Size() - CountOnes();
		}

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

		// ----------------------------------------------------------
		//  Conversion
		// ----------------------------------------------------------
		// Convention matches std::bitset: leftmost character = highest index bit.
		std::string ToString() const
		{
			const std::size_t size = Size();
			std::string s(size, '0');
			for (std::size_t i = 0; i < size; ++i)
			{
				if ((*this)[i])
				{
					s[size - 1 - i] = '1';
				}
			}
			return s;
		}
#endif

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

		template <typename>
		friend class BitsetBase;  // allow ExtractRange/ExtractRangeWithSize

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

		constexpr void SanitizeLastBlock() SEC_NOEXCEPT
		{
			if (const std::size_t size = Size(); size != 0)
			{
				m_Data.back() &= Detail::Bitset::LastBlockMask(size);
			}
		}

	public:
		// ----------------------------------------------------------
		//  Ctors
		// ----------------------------------------------------------
		constexpr Bitset() SEC_NOEXCEPT : m_Data{}
		{
			/* NO CODE */
		}

		constexpr Bitset(const Bitset& other) SEC_NOEXCEPT = default;
		constexpr Bitset(Bitset&& other) SEC_NOEXCEPT = default;
		constexpr Bitset& operator=(const Bitset& other) SEC_NOEXCEPT = default;
		constexpr Bitset& operator=(Bitset&& other) SEC_NOEXCEPT = default;
		~Bitset() SEC_NOEXCEPT = default;

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


	// ============================================================
	//  DynamicBitset  --  runtime-determined size
	// ============================================================
	class DynamicBitset : public BitsetBase<DynamicBitset>
	{
		using Base = BitsetBase;
		friend Base;
		template <typename>
		friend class BitsetBase;  // allow ExtractRange

		constexpr void SanitizeLastBlock() SEC_NOEXCEPT
		{
			if (const std::size_t size = Size(); size != 0)
			{
				m_Data.back() &= Detail::Bitset::LastBlockMask(size);
			}
		}

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
				SanitizeLastBlock();
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
}
