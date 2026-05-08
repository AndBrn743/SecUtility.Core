// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

// #include <SecUtility/Collection/Detail/Bitset.Forward.hpp>
// #include <SecUtility/Collection/Detail/Bitset.Utility.hpp>
// #include <SecUtility/Collection/Detail/BitsetBase.hpp>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winvalid-constexpr"
#endif

namespace SecUtility
{
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


	template <typename Nested>
	class Detail::Bitset::BitsetSegmentExpr : public BitsetBase<BitsetSegmentExpr<Nested>>
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
	struct Traits<Detail::Bitset::BitsetSegmentExpr<Nested>>
	{
		static constexpr bool IsNestedByRef = false;
		using EvaluatedType = DynamicBitset;
	};


	template <typename Nested>
	class Detail::Bitset::BitsetNotExpr : public BitsetBase<BitsetNotExpr<Nested>>
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
	struct Traits<Detail::Bitset::BitsetNotExpr<Nested>>
	{
		static constexpr bool IsNestedByRef = false;
		using EvaluatedType = typename Traits<std::remove_const_t<Nested>>::EvaluatedType;
	};


	template <typename Op, typename Lhs, typename Rhs>
	class Detail::Bitset::BitsetBinaryExpr : public BitsetBase<BitsetBinaryExpr<Op, Lhs, Rhs>>
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
	struct Traits<Detail::Bitset::BitsetBinaryExpr<Op, Lhs, Rhs>>
	{
		static constexpr bool IsNestedByRef = false;
		using EvaluatedType = std::conditional_t<Detail::Bitset::is_fixed_size_bitset<std::decay_t<Rhs>>::value,
		                                         typename Traits<std::remove_const_t<Rhs>>::EvaluatedType,
		                                         typename Traits<std::remove_const_t<Lhs>>::EvaluatedType>;
	};


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
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winvalid-constexpr"
#endif
