//
// Created by Andy on 10/1/2024.
//

#pragma once

#include <tuple>
#include <array>
#include <type_traits>
#include <utility>


namespace SecUtility::ParameterPackUtility
{
	namespace Detail
	{
		template <std::size_t Index>
		struct ParameterExtractor;

		template <>
		struct ParameterExtractor<0>
		{
			template <typename Arg0, typename... Args>
			constexpr static decltype(auto) Extract(Arg0&& arg0, Args&&...) noexcept
			{
				return std::forward<decltype(arg0)>(arg0);
			}
		};

		template <std::size_t Index>
		struct ParameterExtractor
		{
			template <typename Arg0, typename... Args>
			constexpr static decltype(auto) Extract(Arg0&&, Args&&... args) noexcept
			{
				return ParameterExtractor<Index - 1>::Extract(std::forward<decltype(args)>(args)...);
			}
		};


		template <std::size_t Index>
		struct type_at_index;

		template <>
		struct type_at_index<0>
		{
		private:
			// Note: we do need a helper struct because we cannot discard args when defining type aliases
			template <typename Type0, typename...>
			struct type_helper
			{
				using type = Type0;
			};

		public:
			template <typename Type0, typename... Types>
			using type = typename type_helper<Type0, Types...>::type;
		};

		template <std::size_t Index>
		struct type_at_index
		{
		private:
			// Note: we do need a helper struct because we cannot discard args when defining type aliases
			template <typename, typename... Types>
			struct type_helper
			{
				using type = typename type_at_index<Index - 1>::template type<Types...>;
			};

		public:
			template <typename Arg0, typename... Args>
			using type = typename type_helper<Arg0, Args...>::type;
		};

		template <typename TTarget, typename... Ts>
		struct instance_count_in
		{
			static constexpr std::size_t value = (std::size_t{0} + ... + (std::is_same_v<TTarget, Ts> ? 1 : 0));
		};

		template <typename TTarget, typename... Ts>
		struct instance_indices_in
		{
		private:
			static constexpr std::size_t InstanceCount = instance_count_in<TTarget, Ts...>::value;
			using ResultType = std::array<std::size_t, InstanceCount>;

			template <std::size_t... Is>
			static constexpr ResultType Calculate(std::index_sequence<Is...>)
			{
				ResultType result{};

				std::size_t index = 0;
				std::size_t offset = 0;

				((std::is_same_v<TTarget, Ts> ? result[offset] = index, ++index, ++offset : ++index), ...);

				return result;
			}


		public:
			static constexpr ResultType value = Calculate(std::make_index_sequence<InstanceCount>{});
		};

		template <typename...>
		struct has_duplicate;

		template <>
		struct has_duplicate<> : std::false_type
		{
			/* NO CODE */
		};

		template <typename T>
		struct has_duplicate<T> : std::false_type
		{
			/* NO CODE */
		};

		template <typename T, typename... Ts>
		struct has_duplicate<T, Ts...>
		{
			static constexpr bool value = (std::is_same_v<T, Ts> || ...) || has_duplicate<Ts...>::value;
		};

		template <typename...>
		struct Unique;

		template <>
		struct Unique<>
		{
			using type = std::tuple<>;
		};

		template <typename T>
		struct Unique<T>
		{
			using type = std::tuple<T>;
		};

		template <typename First, typename... Rest>
		struct Unique<First, Rest...>
		{
		private:
			template <typename T, typename... TRest>
			static constexpr bool ExistsInRest()
			{
				return (std::is_same_v<T, TRest> || ...);
			}

		public:
			using type =
			        std::conditional_t<ExistsInRest<First, Rest...>(),
			                           typename Unique<Rest...>::type,
			                           decltype(std::tuple_cat(std::tuple<First>{}, typename Unique<Rest...>::type{}))>;
		};
	} 


	template <std::size_t Index, typename Arg0, typename... Args>
	constexpr decltype(auto) Get(Arg0&& arg0, Args&&... args) noexcept
	{
		return Detail::ParameterExtractor<Index>::Extract(std::forward<decltype(arg0)>(arg0),
		                                                  std::forward<decltype(args)>(args)...);
	}

	template <std::size_t Index, typename... Types>
	using TypeAt = typename Detail::type_at_index<Index>::template type<Types...>;

	template <typename TTarget, typename... Ts>
	inline constexpr std::size_t InstanceCountIn = Detail::instance_count_in<TTarget, Ts...>::value;

	template <typename TTarget, typename... Ts>
	inline constexpr auto InstanceIndicesIn = Detail::instance_indices_in<TTarget, Ts...>::value;

	template <typename TTarget, typename... Ts>
	inline constexpr std::enable_if_t<InstanceCountIn<TTarget, Ts...> == 1, std::size_t> InstanceIndexIn =
	        InstanceIndicesIn<TTarget, Ts...>[0];

	template <typename... Ts>
	inline constexpr bool HasDuplicate = Detail::has_duplicate<Ts...>::value;

	/// <summary>
	/// A tuple which each of its element have unique types from the given type parameter pack.
	/// The unique types are selected from the right to left.
	/// </summary>
	template <typename... Ts>
	using UniqueTypeTuple = typename Detail::Unique<Ts...>::type;
}
