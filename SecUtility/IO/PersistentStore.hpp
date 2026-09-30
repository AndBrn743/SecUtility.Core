// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>


namespace SecUtility::IO
{
	template <typename T>
	struct PersistentTraits;

	namespace PersistentStoreDetail
	{
		class StoreState;

		template <typename T, typename = void>
		struct has_leased_type : std::false_type
		{
			/* NO CODE */
		};

		template <typename T>
		struct has_leased_type<T, std::void_t<typename PersistentTraits<T>::LeasedType>> : std::true_type
		{
			/* NO CODE */
		};

		template <typename T>
		inline constexpr bool HasLeasedType = has_leased_type<T>::value;
	}

	class PersistentStore final
	{
	public:
		static PersistentStore OpenForReadOnly(const std::filesystem::path& path);
		static PersistentStore OpenForReadWrite(const std::filesystem::path& path);
		static PersistentStore Create(const std::filesystem::path& path);
		static PersistentStore CreateIfNotExist(const std::filesystem::path& path);

		PersistentStore(PersistentStore&&) noexcept = default;
		PersistentStore& operator=(PersistentStore&&) noexcept = default;
		PersistentStore(const PersistentStore&) = delete;
		PersistentStore& operator=(const PersistentStore&) = delete;
		~PersistentStore() = default;

		template <typename T, typename TValue>
		void Insert(std::string_view key, TValue&& value);

		template <typename T, typename TValue>
		void InsertOrReassign(std::string_view key, TValue&& value);

		template <typename T, typename TValue>
		void Reassign(std::string_view key, TValue&& value);

		template <typename TValue>
		void Insert(std::string_view key, TValue&& value);

		template <typename TValue>
		void InsertOrReassign(std::string_view key, TValue&& value);

		template <typename TValue>
		void Reassign(std::string_view key, TValue&& value);

		template <typename T>
		T Get(std::string_view key) const;

		template <typename T, std::enable_if_t<PersistentStoreDetail::HasLeasedType<T>, int> = 0>
		auto GetLeased(std::string_view key) const -> typename PersistentTraits<T>::LeasedType;

		bool Erase(std::string_view key);
		bool Contains(std::string_view key) const;
		std::size_t Size() const;

	private:
		friend class PersistentStoreDetail::StoreState;

		enum class AccessState
		{
			ReadOnly,
			ReadWrite
		};

		explicit PersistentStore(std::shared_ptr<PersistentStoreDetail::StoreState> statePtr) noexcept
		    : m_StatePtr(std::move(statePtr))
		{
			/* NO CODE */
		}

		std::shared_ptr<PersistentStoreDetail::StoreState> m_StatePtr;
	};
}
