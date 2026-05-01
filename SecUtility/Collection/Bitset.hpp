// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

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

		constexpr std::size_t BlocksFor(const std::size_t bits) noexcept
		{
			return (bits + BitsPerBlock - 1) / BitsPerBlock;
		}

		// Mask of the active bits in the last block.  Returns ~0ull when count % 64 == 0.
		constexpr std::uint64_t LastBlockMask(const std::size_t count) noexcept
		{
			const std::size_t remainder = count % BitsPerBlock;
			return remainder == 0 ? ~std::uint64_t{0} : (std::uint64_t{1} << remainder) - 1u;
		}
	}


	// ============================================================
	//  BitsetBase<Derived>
	// ============================================================
	template <typename Derived>
	class BitsetBase
	{
	public:
		// ----------------------------------------------------------
		//  Proxy reference
		// ----------------------------------------------------------
		class BitReference
		{
			friend BitsetBase;
			constexpr BitReference(std::uint64_t* block, const std::uint64_t mask) noexcept
			    : m_BlockPtr(block), m_Mask(mask)
			{
				/* NO CODE */
			}

		public:
			constexpr BitReference& operator=(const bool value) noexcept
			{
				if (value)
				{
					*m_BlockPtr |= m_Mask;
				}
				else
				{
					*m_BlockPtr &= ~m_Mask;
				}
				return *this;
			}

			constexpr BitReference& operator=(const BitReference& o) noexcept
			{
				return *this = static_cast<bool>(o);
			}

			template <typename OtherDerived>
			constexpr BitReference& operator=(const typename BitsetBase<OtherDerived>::BitReference& o) noexcept
			{
				return *this = static_cast<bool>(o);
			}

			// ReSharper disable once CppNonExplicitConversionOperator
			constexpr operator bool() const noexcept
			{
				return (*m_BlockPtr & m_Mask) != 0;
			}

			// ReSharper disable once CppMemberFunctionMayBeConst
			constexpr void Flip() noexcept
			{
				*m_BlockPtr ^= m_Mask;
			}

		private:
			std::uint64_t* m_BlockPtr;
			std::uint64_t m_Mask;
		};

	private:
		// ----------------------------------------------------------
		//  CRTP plumbing (private, friend Derived only)
		// ----------------------------------------------------------
		friend Derived;

		template <std::size_t>
		friend class Bitset;  // `friend`ship required for cross construct and assign

		friend class DynamicBitset;  // `friend`ship required for cross construct and assign

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
		constexpr std::size_t BlockCount() const noexcept
		{
			return AsDerived().BlockCount();
		}

		constexpr std::uint64_t* Data() noexcept
		{
			return AsDerived().Data();
		}

		constexpr const std::uint64_t* Data() const noexcept
		{
			return AsDerived().Data();
		}

		constexpr void SanitizeLastBlock() SEC_NOEXCEPT
		{
			const std::size_t sz = Size();
			if (sz == 0)
			{
				return;
			}
			Data()[BlockCount() - 1] &= Detail::Bitset::LastBlockMask(sz);
		}

	public:
		// ----------------------------------------------------------
		//  Size / capacity
		// ----------------------------------------------------------
		constexpr std::size_t Size() const noexcept
		{
			return AsDerived().Size();
		}

		constexpr bool IsEmpty() const noexcept
		{
			return Size() == 0;
		}

		// ----------------------------------------------------------
		//  Element access
		// ----------------------------------------------------------
		bool operator[](const std::size_t index) const SEC_NOEXCEPT
		{
			SEC_ASSERT(index < Size());
			return (Data()[index / Detail::Bitset::BitsPerBlock] >> (index % Detail::Bitset::BitsPerBlock)) & 1u;
		}

		constexpr BitReference operator[](const std::size_t index) SEC_NOEXCEPT
		{
			SEC_ASSERT(index < Size());
			const std::uint64_t mask = std::uint64_t{1} << (index % Detail::Bitset::BitsPerBlock);
			return BitReference(&Data()[index / Detail::Bitset::BitsPerBlock], mask);
		}

		bool IsOne(std::size_t index) const SEC_NOEXCEPT
		{
			return (*this)[index];
		}

		bool IsZero(std::size_t index) const SEC_NOEXCEPT
		{
			return !(*this)[index];
		}

		Derived& Set(const std::size_t index, const bool value = true) SEC_NOEXCEPT
		{
			SEC_ASSERT(index < Size());
			const std::size_t b = index / Detail::Bitset::BitsPerBlock;
			const std::uint64_t m = std::uint64_t{1} << (index % Detail::Bitset::BitsPerBlock);
			if (value)
			{
				Data()[b] |= m;
			}
			else
			{
				Data()[b] &= ~m;
			}

			return AsDerived();
		}

		Derived& Reset(const std::size_t index) SEC_NOEXCEPT
		{
			return Set(index, false);
		}

		Derived& Flip(const std::size_t index) SEC_NOEXCEPT
		{
			SEC_ASSERT(index < Size());
			Data()[index / Detail::Bitset::BitsPerBlock] ^= std::uint64_t{1} << (index % Detail::Bitset::BitsPerBlock);
			return AsDerived();
		}

		// ----------------------------------------------------------
		//  Comparison
		// ----------------------------------------------------------
		template <typename OtherDerived>
		bool operator==(const BitsetBase<OtherDerived>& other) const SEC_NOEXCEPT
		{
			if (Size() != other.Size())
			{
				return false;
			}
			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				if (Data()[i] != other.Data()[i])
				{
					return false;
				}
			}
			return true;
		}

		template <typename OtherDerived>
		bool operator!=(const BitsetBase<OtherDerived>& other) const SEC_NOEXCEPT
		{
			return !(*this == other);
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

		template <typename>
		friend class BitsetBase;  // allow ExtractRange/ExtractRangeWithSize

		static constexpr std::size_t kN = N;
		static constexpr std::size_t kBlockCount =
		        (N + Detail::Bitset::BitsPerBlock - 1) / Detail::Bitset::BitsPerBlock;

	public:
		// ----------------------------------------------------------
		//  Ctors
		// ----------------------------------------------------------
		constexpr Bitset() noexcept : m_Data{}
		{
			/* NO CODE */
		}

		constexpr Bitset(const Bitset& other) noexcept = default;
		constexpr Bitset(Bitset&& other) noexcept = default;
		constexpr Bitset& operator=(const Bitset& other) noexcept = default;
		constexpr Bitset& operator=(Bitset&& other) noexcept = default;
		~Bitset() noexcept = default;

		template <typename OtherDerived>
		explicit Bitset(const BitsetBase<OtherDerived>& other) : m_Data{}
		{
			SEC_ASSERT(other.Size() == N);
			for (std::size_t i = 0; i < kBlockCount; ++i)
			{
				m_Data[i] = other.Data()[i];
			}
		}

		template <typename OtherDerived>
		constexpr Bitset& operator=(const BitsetBase<OtherDerived>& other)
		{
			SEC_ASSERT(other.Size() == N);
			for (std::size_t i = 0; i < kBlockCount; ++i)
			{
				m_Data[i] = other.Data()[i];
			}
			return *this;
		}

		template <typename OtherDerived>
		constexpr Bitset& operator=(BitsetBase<OtherDerived>&& other)
		{
			return *this = static_cast<const BitsetBase<OtherDerived>&>(other);
		}

		// ----------------------------------------------------------
		//  Size / capacity
		// ----------------------------------------------------------
		constexpr std::size_t Size() const noexcept
		{
			return kN;
		}

	private:
		constexpr std::size_t BlockCount() const noexcept
		{
			return kBlockCount;
		}

		std::uint64_t* Data() noexcept
		{
			return m_Data;
		}

		const std::uint64_t* Data() const noexcept
		{
			return m_Data;
		}

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

		explicit DynamicBitset(const std::size_t size, const bool value = false)
		    : m_Size(size), m_Data(Detail::Bitset::BlocksFor(size), value ? ~std::uint64_t{0} : std::uint64_t{0})
		{
			if (value)
			{
				SanitizeLastBlock();
			}
		}

		// ----------------------------------------------------------
		//  Size / capacity
		// ----------------------------------------------------------
		constexpr std::size_t Size() const noexcept
		{
			return m_Size;
		}

	private:
		constexpr std::size_t BlockCount() const noexcept
		{
			return m_Data.size();
		}

		std::uint64_t* Data() noexcept
		{
			return m_Data.data();
		}

		const std::uint64_t* Data() const noexcept
		{
			return m_Data.data();
		}

		std::size_t m_Size;
		std::vector<std::uint64_t> m_Data;
	};
}
