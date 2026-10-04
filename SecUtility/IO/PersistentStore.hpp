// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#include <SecUtility/IO/PersistentStore/EncodingId.hpp>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>


namespace SecUtility::IO
{
	template <typename T, typename = void>
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

		template <typename T = void, typename TValue>
		void Insert(std::string_view key, TValue&& value);

		template <typename T = void, typename TValue>
		void InsertOrReassign(std::string_view key, TValue&& value);

		template <typename T = void, typename TValue>
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

		explicit PersistentStore(std::shared_ptr<PersistentStoreDetail::StoreState> statePtr) noexcept
		    : m_StatePtr(std::move(statePtr))
		{
			/* NO CODE */
		}

		std::shared_ptr<PersistentStoreDetail::StoreState> m_StatePtr;
	};
}

#include <SecUtility/IO/PersistentStore/Detail/StoreState.hpp>

namespace SecUtility::IO
{
	inline PersistentStore PersistentStore::OpenForReadOnly(const std::filesystem::path& path)
	{
		return PersistentStore(PersistentStoreDetail::StoreState::Open(
		        path, PersistentStoreDetail::FileAccess::ReadOnly));
	}

	inline PersistentStore PersistentStore::OpenForReadWrite(const std::filesystem::path& path)
	{
		return PersistentStore(PersistentStoreDetail::StoreState::Open(
		        path, PersistentStoreDetail::FileAccess::ReadWrite));
	}

	inline PersistentStore PersistentStore::Create(const std::filesystem::path& path)
	{
		return PersistentStore(PersistentStoreDetail::StoreState::Create(path));
	}

	inline PersistentStore PersistentStore::CreateIfNotExist(const std::filesystem::path& path)
	{
		return PersistentStore(PersistentStoreDetail::StoreState::CreateIfNotExist(path));
	}

	inline bool PersistentStore::Contains(const std::string_view key) const
	{
		return m_StatePtr->Contains(key);
	}

	inline std::size_t PersistentStore::Size() const
	{
		return m_StatePtr->Size();
	}

	// ReSharper disable once CppMemberFunctionMayBeConst
	inline bool PersistentStore::Erase(const std::string_view key)
	{
		return m_StatePtr->Erase(key);
	}

	template <typename T, typename TValue>
	void PersistentStore::Insert(const std::string_view key, TValue&& value)
	{
		using StoredType = std::conditional_t<std::is_void_v<T>, std::decay_t<TValue>, T>;
		m_StatePtr->template Insert<StoredType>(key, std::forward<TValue>(value));
	}

	template <typename T, typename TValue>
	void PersistentStore::InsertOrReassign(const std::string_view key, TValue&& value)
	{
		using StoredType = std::conditional_t<std::is_void_v<T>, std::decay_t<TValue>, T>;
		m_StatePtr->template InsertOrReassign<StoredType>(key, std::forward<TValue>(value));
	}

	template <typename T, typename TValue>
	void PersistentStore::Reassign(const std::string_view key, TValue&& value)
	{
		using StoredType = std::conditional_t<std::is_void_v<T>, std::decay_t<TValue>, T>;
		m_StatePtr->template Reassign<StoredType>(key, std::forward<TValue>(value));
	}

	template <typename T>
	T PersistentStore::Get(const std::string_view key) const
	{
		return m_StatePtr->template Get<T>(key);
	}

	template <typename T, std::enable_if_t<PersistentStoreDetail::HasLeasedType<T>, int>>
	auto PersistentStore::GetLeased(const std::string_view key) const -> typename PersistentTraits<T>::LeasedType
	{
		return m_StatePtr->template GetLeased<T>(key);
	}
}
