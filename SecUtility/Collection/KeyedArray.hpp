// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#if __cplusplus >= 202002L

#include <SecUtility/Diagnostic/Exception.hpp>
#include <SecUtility/Meta/ParameterPackUtility.hpp>
#include <SecUtility/Meta/TypeTrait.hpp>
#include <array>
#include <cstddef>
#include <functional>
#include <utility>


namespace SecUtility
{
	namespace Detail::KeyedArray
	{
		// linear search: returns the index of `needle` in the key list, or N (sentinel) when not found
		template <typename Key, std::size_t N>
		constexpr std::size_t IndexOf(const std::array<Key, N>& keys, const Key& needle) noexcept
		{
			for (std::size_t i = 0; i < N; ++i)
			{
				if (keys[i] == needle)
				{
					return i;
				}
			}
			return N;  // not found
		}

		template <typename Key, std::size_t N>
		consteval bool IsAllUnique(const std::array<Key, N>& keys) noexcept
		{
			for (std::size_t i = 0; i < N; ++i)
			{
				for (std::size_t j = i + 1; j < N; ++j)
				{
					if (keys[i] == keys[j])
					{
						return false;
					}
				}
			}
			return true;
		}

		template <auto... Keys>
		consteval bool IsAllUnique() noexcept
		{
			return IsAllUnique(std::array<std::common_type_t<decltype(Keys)...>, sizeof...(Keys)>{Keys...});
		}
	}

	template <typename Value, auto... Keys>
	class KeyedArray
	{
		static constexpr std::size_t KSize = sizeof...(Keys);

	public:
		using KeyType = std::common_type_t<decltype(Keys)...>;
		using ValueType = Value;

		static constexpr std::array<KeyType, KSize> KeySet = {Keys...};

		static_assert(Detail::KeyedArray::IsAllUnique(KeySet), "KeyedArray: duplicate keys are not allowed.");

		template <KeyType K>
		static constexpr bool Contains = Detail::KeyedArray::IndexOf(KeySet, K) < KSize;

		constexpr KeyedArray() = default;

		constexpr explicit KeyedArray(std::array<Value, KSize> values) noexcept : m_Data(std::move(values))
		{
			/* NO CODE */
		}

		template <typename... ConvertableToValueType>
		constexpr explicit KeyedArray(ConvertableToValueType&&... values)  //
		        noexcept((noexcept(static_cast<ValueType>(std::forward<ConvertableToValueType>(values))) && ...))
		    requires(sizeof...(ConvertableToValueType) == KSize
		             && (std::is_convertible_v<ConvertableToValueType, ValueType> && ...))
		    : m_Data{static_cast<ValueType>(std::forward<ConvertableToValueType>(values))...}
		{
			/* NO CODE */
		}

		template <KeyType K>
		[[nodiscard]] constexpr Value& Get() noexcept
		{
			static_assert(Contains<K>, "KeyedArray::Get<K>(): key K is not in this KeyedArray.");
			return m_Data[Detail::KeyedArray::IndexOf(KeySet, K)];
		}

		template <KeyType K>
		[[nodiscard]] constexpr const Value& Get() const noexcept
		{
			static_assert(Contains<K>, "KeyedArray::Get<K>(): key K is not in this KeyedArray.");
			return m_Data[Detail::KeyedArray::IndexOf(KeySet, K)];
		}

		[[nodiscard]] constexpr Value& operator[](const KeyType& key)
		{
			const std::size_t index = Detail::KeyedArray::IndexOf(KeySet, key);
			if (index == KSize)
			{
				throw ArgumentOutOfRangeException("KeyedArray::operator[]: key not found.");
			}
			return m_Data[index];
		}

		[[nodiscard]] constexpr const Value& operator[](const KeyType& key) const
		{
			const std::size_t index = Detail::KeyedArray::IndexOf(KeySet, key);
			if (index == KSize)
			{
				throw ArgumentOutOfRangeException("KeyedArray::operator[]: key not found.");
			}
			return m_Data[index];
		}

		template <typename Fn>
		constexpr void Foreach(Fn&& f)
		{
			if constexpr (std::invocable<Fn, const KeyType&, Value&>)
			{
				[&]<std::size_t... I>(std::index_sequence<I...>)
				{ (f(KeySet[I], m_Data[I]), ...); }(std::make_index_sequence<KSize>{});
			}
			else if constexpr (std::invocable<Fn, const KeyType&, Value&, std::size_t>)
			{
				[&]<std::size_t... I>(std::index_sequence<I...>)
				{ (f(KeySet[I], m_Data[I], I), ...); }(std::make_index_sequence<KSize>{});
			}
			else
			{
				auto it = m_Data.begin();
				((f.template operator()<Keys>(*it), ++it), ...);
			}
		}

		template <typename Fn>
		constexpr void Foreach(Fn&& f) const
		{
			if constexpr (std::invocable<Fn, const KeyType&, const Value&>)
			{
				[&]<std::size_t... I>(std::index_sequence<I...>)
				{ (f(KeySet[I], m_Data[I]), ...); }(std::make_index_sequence<KSize>{});
			}
			else if constexpr (std::invocable<Fn, const KeyType&, const Value&, std::size_t>)
			{
				[&]<std::size_t... I>(std::index_sequence<I...>)
				{ (f(KeySet[I], m_Data[I], I), ...); }(std::make_index_sequence<KSize>{});
			}
			else
			{
				auto it = m_Data.begin();
				((f.template operator()<Keys>(*it), ++it), ...);
			}
		}

		[[nodiscard]] static constexpr std::size_t Size() noexcept
		{
			return KSize;
		}

		[[nodiscard]] constexpr Value* data() noexcept
		{
			return m_Data.data();
		}

		[[nodiscard]] constexpr const Value* data() const noexcept
		{
			return m_Data.data();
		}

		[[nodiscard]] constexpr bool operator==(const KeyedArray&) const = default;

	private:
		std::array<Value, KSize> m_Data{};
	};

	namespace Detail::KeyedArray
	{
		template <typename>
		struct is_keyed_array : std::false_type
		{
			/* NO CODE */
		};

		template <typename V, auto... Ks>
		struct is_keyed_array<SecUtility::KeyedArray<V, Ks...>> : std::true_type
		{
			/* NO CODE */
		};

		template <typename...>
		struct concatenated_type;  // concatenated

		template <typename V, auto... Ks>
		struct concatenated_type<SecUtility::KeyedArray<V, Ks...>>
		{
			using type = SecUtility::KeyedArray<V, Ks...>;
		};

		template <typename V, auto... Ks, typename W, auto... Ls>
		struct concatenated_type<SecUtility::KeyedArray<V, Ks...>, SecUtility::KeyedArray<W, Ls...>>
		{
			using type = SecUtility::KeyedArray<std::common_type_t<V, W>, Ks..., Ls...>;
		};

		template <typename KA0, typename KA1, typename... KAs>
		struct concatenated_type<KA0, KA1, KAs...>
		{
			using type = concatenated_type<typename concatenated_type<KA0, KA1>::type, KAs...>::type;
		};

		template <typename T, T Value>
		struct constant
		{
			static constexpr T value = Value;
			using type = T;
		};

		template <typename... Args>
		struct merged_type;

		template <typename Value, auto... Keys, typename OtherValue, auto... OtherKeys>
		struct merged_type<SecUtility::KeyedArray<Value, Keys...>, SecUtility::KeyedArray<OtherValue, OtherKeys...>>
		{
		private:
			using ValueType = std::common_type_t<Value, OtherValue>;
			using KeyType = std::common_type_t<decltype(Keys)..., decltype(OtherKeys)...>;
			using KeyTypeTuple =
			        ParameterPackUtility::UniqueTypeTuple<constant<KeyType, Keys>..., constant<KeyType, OtherKeys>...>;

			template <typename ktt>
			struct result_type;

			template <typename... VKeys>
			struct result_type<std::tuple<VKeys...>>
			{
				using type = SecUtility::KeyedArray<ValueType, VKeys::value...>;
			};

		public:
			using type = result_type<KeyTypeTuple>::type;
		};

		template <typename Value,
		          auto... Keys,
		          typename OtherValue,
		          auto... OtherKeys,
		          typename... OtherOtherKeyedArrays>
		struct merged_type<SecUtility::KeyedArray<Value, Keys...>,
		                   SecUtility::KeyedArray<OtherValue, OtherKeys...>,
		                   OtherOtherKeyedArrays...>
		{
			using type = merged_type<typename merged_type<SecUtility::KeyedArray<Value, Keys...>,
			                                              SecUtility::KeyedArray<OtherValue, OtherKeys...>>::type,
			                         OtherOtherKeyedArrays...>::type;
		};
	}

	template <typename... KeyedArrays>
	constexpr auto Concat(KeyedArrays&&... arrays)
	    requires((Detail::KeyedArray::is_keyed_array<std::remove_cvref_t<KeyedArrays>>::value && ...))
	{
		using Result = Detail::KeyedArray::concatenated_type<std::remove_cvref_t<KeyedArrays>...>::type;
		Result result;

		auto copyOrMoveToResult = [&result, offset = std::size_t{0}]<typename Source>(Source&& source) mutable
		{
			for (std::size_t i = 0; i < std::remove_cvref_t<Source>::Size(); ++i, ++offset)
			{
				if constexpr (std::is_lvalue_reference_v<Source>)
				{
					result.data()[offset] = source.data()[i];
				}
				else
				{
					result.data()[offset] = std::move(source.data()[i]);
				}
			}
		};

		(copyOrMoveToResult(std::forward<KeyedArrays>(arrays)), ...);

		return result;
	}

	// function that matches the following signature is reserved by SecUtility for future use
	template <typename Op, typename Init, typename... KeyedArrays>
	constexpr auto Merge(Op op, Init init, KeyedArrays&&... arrays)
	    requires((Detail::KeyedArray::is_keyed_array<std::remove_cvref_t<KeyedArrays>>::value && ...))
	= delete;
}

#endif
