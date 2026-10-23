// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if !defined(SEC_BITSET_DETAIL)
#error "Please do not directly include internal headers"
#endif

namespace SecUtility
{
	// Reads logical words without realignment. UnpaddedBlock(i) supplies a valid
	// logical word; readers mask unused bits in the last word. This guarantees
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
