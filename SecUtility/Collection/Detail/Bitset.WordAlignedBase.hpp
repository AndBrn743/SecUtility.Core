// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if !defined(SEC_BITSET_DETAIL)
#error "Please do not directly include internal headers"
#endif

namespace SecUtility
{
	// Reads logical words supplied by UnpaddedBlock(i), including expressions
	// whose view leaves need realignment. Each read supplies a valid
	// logical word; readers ignore unused bits in the last word. This guarantees
	// neither contiguous storage nor cleared tail padding, and says nothing about
	// the physical alignment used by Block() or AlignedTo_Impl().
	template <typename Derived>
	class WordAlignedBitsetBase : public BitsetBase<Derived>
	{
		using Base = BitsetBase<Derived>;
		friend Derived;
		friend Base;
		friend OwningBitsetBase<Derived>;

		constexpr WordAlignedBitsetBase() noexcept = default;
		constexpr WordAlignedBitsetBase(const WordAlignedBitsetBase&) noexcept = default;
		constexpr WordAlignedBitsetBase(WordAlignedBitsetBase&&) noexcept = default;
		constexpr WordAlignedBitsetBase& operator=(const WordAlignedBitsetBase&) noexcept = default;
		constexpr WordAlignedBitsetBase& operator=(WordAlignedBitsetBase&&) noexcept = default;
		~WordAlignedBitsetBase() noexcept = default;

		using Base::UnpaddedBlock;

	public:
		using Base::operator=;
		using Base::AsDerived;
		using Base::Size;

	private:
		template <bool One>
		constexpr std::uint64_t MatchingWord(const std::size_t i) const SEC_NOEXCEPT
		{
			const std::uint64_t word = UnpaddedBlock(i);
			return One ? word : ~word;
		}

		// Inclusive start. A forward scan can encounter unspecified tail bits;
		// clamping its first match also handles a complemented final word.
		template <bool One>
		constexpr std::size_t FindForward(const std::size_t first) const SEC_NOEXCEPT
		{
			const std::size_t size = Size();
			if (first >= size)
			{
				return size;
			}
			std::size_t i = first / Detail::Bitset::BitsPerBlock;
			const std::size_t offset = first % Detail::Bitset::BitsPerBlock;
			if (const std::uint64_t word = MatchingWord<One>(i) >> offset; word != 0)
			{
				return std::min(size, first + Detail::Bitset::TrailingZeroCount(word));
			}
			const std::size_t words = Detail::Bitset::BlocksFor(size);
			for (++i; i < words; ++i)
			{
				if (const std::uint64_t word = MatchingWord<One>(i); word != 0)
				{
					return std::min(size, i * Detail::Bitset::BitsPerBlock + Detail::Bitset::TrailingZeroCount(word));
				}
			}
			return size;
		}

		// Exclusive end, at most Size(). Shifting the first word clears both
		// bits beyond the cursor and any unspecified tail padding.
		template <bool One>
		constexpr std::size_t FindBackward(const std::size_t end) const SEC_NOEXCEPT
		{
			if (end == 0)
			{
				return Size();
			}
			std::size_t i = (end - 1) / Detail::Bitset::BitsPerBlock;
			const std::size_t offset = (end - 1) % Detail::Bitset::BitsPerBlock;
			if (const std::uint64_t word = MatchingWord<One>(i) << (Detail::Bitset::BitsPerBlock - 1 - offset);
			    word != 0)
			{
				return end - 1 - Detail::Bitset::LeadingZeroCount(word);
			}
			while (i-- > 0)
			{
				if (const std::uint64_t word = MatchingWord<One>(i); word != 0)
				{
					return i * Detail::Bitset::BitsPerBlock + Detail::Bitset::BitsPerBlock - 1
					       - Detail::Bitset::LeadingZeroCount(word);
				}
			}
			return Size();
		}

		constexpr std::size_t TrailingZeroCount_Impl() const noexcept
		{
			return TrailingCount<true>();
		}

		constexpr std::size_t TrailingOneCount_Impl() const noexcept
		{
			return TrailingCount<false>();
		}

		constexpr std::size_t LeadingZeroCount_Impl() const noexcept
		{
			return LeadingCount<true>();
		}

		constexpr std::size_t LeadingOneCount_Impl() const noexcept
		{
			return LeadingCount<false>();
		}

		// Run counts have fixed endpoints. Keeping those loops separate from
		// cursor searches exposes their bounds and avoids index/count conversions.
		template <bool One>
		constexpr std::size_t TrailingCount() const noexcept
		{
			const std::size_t size = Size();
			const std::size_t words = Detail::Bitset::BlocksFor(size);
			for (std::size_t i = 0; i < words; ++i)
			{
				if (const std::uint64_t word = MatchingWord<One>(i); word != 0)
				{
					return std::min(size, i * Detail::Bitset::BitsPerBlock + Detail::Bitset::TrailingZeroCount(word));
				}
			}
			return size;
		}

		template <bool One>
		constexpr std::size_t LeadingCount() const noexcept
		{
			const std::size_t size = Size();
			if (size == 0)
			{
				return 0;
			}
			const std::size_t last = Detail::Bitset::BlocksFor(size) - 1;
			const std::size_t padding = (last + 1) * Detail::Bitset::BitsPerBlock - size;
			if (const std::uint64_t word = MatchingWord<One>(last) << padding; word != 0)
			{
				return Detail::Bitset::LeadingZeroCount(word);
			}
			for (std::size_t i = last; i-- > 0;)
			{
				if (const std::uint64_t word = MatchingWord<One>(i); word != 0)
				{
					return size - (i + 1) * Detail::Bitset::BitsPerBlock + Detail::Bitset::LeadingZeroCount(word);
				}
			}
			return size;
		}

		constexpr std::size_t IndexOfNextOne_Impl(const std::size_t pos) const SEC_NOEXCEPT
		{
			const std::size_t size = Size();
			return size == 0 || pos >= size - 1 ? size : FindForward<true>(pos + 1);
		}

		constexpr std::size_t IndexOfNextZero_Impl(const std::size_t pos) const SEC_NOEXCEPT
		{
			const std::size_t size = Size();
			return size == 0 || pos >= size - 1 ? size : FindForward<false>(pos + 1);
		}

		constexpr std::size_t IndexOfPreviousOne_Impl(const std::size_t pos) const SEC_NOEXCEPT
		{
			return FindBackward<true>(pos);
		}

		constexpr std::size_t IndexOfPreviousZero_Impl(const std::size_t pos) const SEC_NOEXCEPT
		{
			return FindBackward<false>(pos);
		}

		// Public predicate wrappers handle empty bitsets before reaching these hooks.
		constexpr bool IsAllOnes_Impl() const noexcept
		{
			const std::size_t size = Size();
			const std::size_t last = Detail::Bitset::BlocksFor(size) - 1;
			for (std::size_t i = 0; i < last; ++i)
			{
				if (UnpaddedBlock(i) != ~std::uint64_t{0})
				{
					return false;
				}
			}
			const std::uint64_t mask = Detail::Bitset::TailMask(size);
			return (UnpaddedBlock(last) & mask) == mask;
		}

		constexpr bool IsAllZeros_Impl() const noexcept
		{
			const std::size_t size = Size();
			const std::size_t last = Detail::Bitset::BlocksFor(size) - 1;
			for (std::size_t i = 0; i < last; ++i)
			{
				if (UnpaddedBlock(i) != 0)
				{
					return false;
				}
			}
			return (UnpaddedBlock(last) & Detail::Bitset::TailMask(size)) == 0;
		}

		constexpr std::size_t OneCount_Impl() const noexcept
		{
			const std::size_t size = Size();
			const std::size_t words = Detail::Bitset::BlocksFor(size);
			if (words == 0)
			{
				return 0;
			}
			std::size_t count = 0;
			for (std::size_t i = 0; i < words - 1; ++i)
			{
				count += Detail::Bitset::PopCount(UnpaddedBlock(i));
			}
			return count + Detail::Bitset::PopCount(UnpaddedBlock(words - 1) & Detail::Bitset::TailMask(size));
		}

		template <typename OtherDerived>
		constexpr bool Equal_Impl(const BitsetBase<OtherDerived>& other) const SEC_NOEXCEPT
		{
			if constexpr (std::is_base_of_v<WordAlignedBitsetBase<OtherDerived>, OtherDerived>)
			{
				const std::size_t size = Size();
				const std::size_t words = Detail::Bitset::BlocksFor(size);
				if (words == 0)
				{
					return true;
				}
				for (std::size_t i = 0; i < words - 1; ++i)
				{
					if (UnpaddedBlock(i) != other.UnpaddedBlock(i))
					{
						return false;
					}
				}
				return ((UnpaddedBlock(words - 1) ^ other.UnpaddedBlock(words - 1)) & Detail::Bitset::TailMask(size))
				       == 0;
			}
			else
			{
				return Base::Equal_Impl(other);
			}
		}
	};

	namespace Detail::Bitset
	{
		template <typename Derived, typename... Operands>
		using BitsetExpressionBase = std::conditional_t<
		        (std::is_base_of_v<WordAlignedBitsetBase<std::decay_t<Operands>>, std::decay_t<Operands>> && ...),
		        WordAlignedBitsetBase<Derived>,
		        BitsetBase<Derived>>;
	}
}
