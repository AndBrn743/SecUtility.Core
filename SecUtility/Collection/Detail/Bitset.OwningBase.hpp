// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if !defined(SEC_BITSET_DETAIL)
#error "Please do not directly include internal headers"
#endif

namespace SecUtility
{
	// Owns contiguous words, with zero head padding and cleared tail padding.
	// Concrete owners provide storage and size; the common API stays in BitsetBase.
	template <typename Derived>
	class OwningBitsetBase : public WordAlignedBitsetBase<Derived>
	{
		using Base = WordAlignedBitsetBase<Derived>;
		friend Derived;
		friend BitsetBase<Derived>;
		friend Base;
		template <typename>
		friend class OwningBitsetBase;

		constexpr OwningBitsetBase() noexcept = default;
		constexpr OwningBitsetBase(const OwningBitsetBase&) noexcept = default;
		constexpr OwningBitsetBase(OwningBitsetBase&&) noexcept = default;
		constexpr OwningBitsetBase& operator=(const OwningBitsetBase&) noexcept = default;
		constexpr OwningBitsetBase& operator=(OwningBitsetBase&&) noexcept = default;
		~OwningBitsetBase() noexcept = default;

		using Base::Block;
		using Base::BlockCount;
		using Base::RestPaddingBitsOfLastBlock;

	public:
		using Base::operator=;
		using Base::AsDerived;
		using Base::Size;

	private:
		constexpr const std::uint64_t* OwningData() const noexcept
		{
			return AsDerived().Data();
		}

		constexpr std::uint64_t UnpaddedBlock_Impl(const std::size_t i) const SEC_NOEXCEPT
		{
			return Block(i);
		}

		constexpr std::size_t OneCount_Impl() const noexcept
		{
			// Cleared padding permits a single reduction over the storage words.
			std::size_t count = 0;
			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				count += Detail::Bitset::PopCount(Block(i));
			}
			return count;
		}

		// Cleared tail padding permits the existing raw zero-run scans.
		constexpr std::size_t TrailingZeroCount_Impl() const noexcept
		{
			return BitsetBase<Derived>::TrailingZeroCount_Impl();
		}

		constexpr std::size_t LeadingZeroCount_Impl() const noexcept
		{
			return BitsetBase<Derived>::LeadingZeroCount_Impl();
		}

		constexpr std::size_t IndexOfNextOne_Impl(const std::size_t pos) const SEC_NOEXCEPT
		{
			// Preserve the raw owning search loop, guarding a wrapped cursor.
			if (pos + 1 == 0)
			{
				return Size();
			}
			return BitsetBase<Derived>::IndexOfNextOne_Impl(pos);
		}

		template <typename OtherDerived>
		constexpr void Assign_Impl(const BitsetBase<OtherDerived>& other) SEC_NOEXCEPT
		{
			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				Block(i) = other.UnpaddedBlock(i);
			}
			RestPaddingBitsOfLastBlock();
		}

		constexpr void SetAll_Impl(const bool value) SEC_NOEXCEPT
		{
			const std::size_t size = Size();
			if (const std::size_t count = Detail::Bitset::BlocksFor(size); count != 0)
			{
				auto* const data = &Block(0);
				const std::uint64_t fill = value ? ~std::uint64_t{0} : std::uint64_t{0};
				for (std::size_t i = 0; i < count - 1; ++i)
				{
					data[i] = fill;
				}
				data[count - 1] = fill & Detail::Bitset::TailMask(size);
			}
		}

		constexpr void FlipAll_Impl() noexcept
		{
			const std::size_t size = Size();
			if (const std::size_t count = Detail::Bitset::BlocksFor(size); count != 0)
			{
				auto* const data = &Block(0);
				for (std::size_t i = 0; i < count - 1; ++i)
				{
					data[i] = ~data[i];
				}
				data[count - 1] = ~data[count - 1] & Detail::Bitset::TailMask(size);
			}
		}

		// Cleared tail padding permits raw comparisons, without a final mask.
		constexpr bool IsAllOnes_Impl() const noexcept
		{
			// Owning storage has no head padding; only the last block can be partial.
			const std::size_t last = BlockCount() - 1;
			for (std::size_t i = 0; i < last; ++i)
			{
				if (Block(i) != ~std::uint64_t{0})
				{
					return false;
				}
			}
			return Block(last) == Detail::Bitset::TailMask(Size());
		}

		constexpr bool IsAllZeros_Impl() const noexcept
		{
			for (std::size_t i = 0; i < BlockCount(); ++i)
			{
				// Owning storage keeps padding clear, so raw zero checks are sufficient.
				const auto block = Block(i);
				if (block != 0)
				{
					return false;
				}
			}

			return true;
		}

		// The public operators handle n == 0 and n >= Size() before entering these kernels.
		constexpr void ShiftLeft_Impl(const std::size_t n) noexcept
		{
			const std::size_t blockShift = n / Detail::Bitset::BitsPerBlock;
			const std::size_t bitShift = n % Detail::Bitset::BitsPerBlock;
			auto* const data = &Block(0);
			const std::size_t count = BlockCount();
			if (bitShift == 0)
			{
				std::copy_backward(data, data + count - blockShift, data + count);
			}
			else
			{
				for (std::size_t i = count - 1; i > blockShift; --i)
				{
					data[i] = (data[i - blockShift] << bitShift)
					          | (data[i - blockShift - 1] >> (Detail::Bitset::BitsPerBlock - bitShift));
				}
				data[blockShift] = data[0] << bitShift;
			}
			std::fill_n(data, blockShift, std::uint64_t{0});
			RestPaddingBitsOfLastBlock();
		}

		constexpr void ShiftRight_Impl(const std::size_t n) noexcept
		{
			const std::size_t blockShift = n / Detail::Bitset::BitsPerBlock;
			const std::size_t bitShift = n % Detail::Bitset::BitsPerBlock;
			auto* const data = &Block(0);
			const std::size_t count = BlockCount();
			const std::size_t limit = count - blockShift - 1;
			if (bitShift == 0)
			{
				if (Detail::Bitset::is_fixed_size_bitset<Derived>::value && count <= 4)
				{
					// A fixed trip count exposes both the retained words and zero fill,
					// allowing small copies and shifts to combine in registers.
					for (std::size_t i = 0; i < count; ++i)
					{
						const std::size_t source = i + blockShift;
						data[i] = source < count ? data[source] : std::uint64_t{0};
					}
					return;
				}
				std::copy(data + blockShift, data + count, data);
			}
			else
			{
				const std::uint64_t last = data[count - 1] >> bitShift;
				for (std::size_t i = 0; i < limit; ++i)
				{
					data[i] = (data[i + blockShift] >> bitShift)
					          | (data[i + blockShift + 1] << (Detail::Bitset::BitsPerBlock - bitShift));
				}
				data[limit] = last;
			}
			std::fill_n(data + limit + 1, blockShift, std::uint64_t{0});
		}

		template <typename OtherDerived>
		constexpr bool Equal_Impl(const BitsetBase<OtherDerived>& other) const SEC_NOEXCEPT
		{
			if constexpr (Traits<OtherDerived>::IsOwning)
			{
				const std::size_t count = BlockCount();
				if (count == 0)
				{
					return true;
				}

#if defined(SEC_IF_NOT_CONSTEVAL)
				SEC_IF_NOT_CONSTEVAL
				{
					// Avoid a library call for small bitsets and first-word mismatches.
					if (count > 4)
					{
						// Owning blocks have no head padding and cleared tail padding.
						// Compare the block arrays, not the enclosing objects.
						const std::size_t bytes = (count - 1) * sizeof(std::uint64_t);
						return Block(0) == other.Block(0)
						       && std::memcmp(OwningData() + 1, other.AsDerived().OwningData() + 1, bytes) == 0;
					}
				}
#endif
				for (std::size_t i = 0; i < count; ++i)
				{
					if (Block(i) != other.Block(i))
					{
						return false;
					}
				}
				return true;
			}
			else
			{
				return Base::Equal_Impl(other);
			}
		}

		template <typename OtherDerived, typename Op>
		Derived& BitwiseCombinationOp_Impl(const BitsetBase<OtherDerived>& other, const Op op) SEC_NOEXCEPT
		{
			if constexpr (Traits<OtherDerived>::IsOwning)
			{
				SEC_ASSERT(Size() == other.Size());
				if (const std::size_t count = BlockCount(); count != 0)
				{
					auto* const data = &Block(0);
					const auto* const source = other.AsDerived().OwningData();
					// AND/OR/XOR preserve cleared padding when both operands own their storage.
					for (std::size_t i = 0; i < count; ++i)
					{
						data[i] = op(data[i], source[i]);
					}
				}
				return AsDerived();
			}
			else
			{
				return Base::BitwiseCombinationOp_Impl(other, op);
			}
		}
	};
}
