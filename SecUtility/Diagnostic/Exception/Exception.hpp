// SPDX-License-Identifier: MIT
// Copyright (c) 2023-2026 Andy Brown

#pragma once

#include <exception>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>


/*

Exception
├── LogicException
│   ├── InvalidArgumentException
│   │   ├── ArgumentNullException
│   │   └── ArgumentOutOfRangeException
│   ├── InvalidOperationException
│   ├── NotImplementedException
│   ├── NotSupportedException
│   ├── UnreachableException  // calls terminate
│	├── PreconditionViolationException
│	├── PostconditionViolationException
│	└── InvariantViolationException
├── RuntimeException
│   ├── IOException
│   ├── TimeoutException
│   ├── OperationCanceledException
│   └── ResourceException
└── ~~AggregateException~~

 */


namespace SecUtility
{
	class Exception : public std::exception
	{
	public:
		Exception() noexcept = default;

		template <typename... Messages>
		explicit Exception(Messages&&... messages) noexcept
		{
			BuildMessage(std::forward<Messages>(messages)...);
		}

		const char* what() const noexcept override
		{
			return m_CMessage;
		}

		Exception(Exception&& other) noexcept : m_Message(std::move(other.m_CMessage)), m_CMessage(m_Message.c_str())
		{
			m_Message = std::move(other.m_Message);
			m_CMessage = m_Message.c_str();
		}

		Exception& operator=(Exception&& other) noexcept
		{
			m_Message = std::move(other.m_Message);
			m_CMessage = m_Message.c_str();
			return *this;
		}

		~Exception() noexcept override = default;

		// create a pr if you do have a need for copying exception objects
		Exception(const Exception&) = delete;
		Exception& operator=(const Exception&) = delete;


	protected:
		static constexpr auto Joiner = "\n\t-> ";

		struct MessageHeaderSuppressor
		{
			/* NO CODE */
		};

		explicit Exception(MessageHeaderSuppressor) noexcept : Exception()
		{
			/* NO CODE */
		}

		template <typename... Messages>
		explicit Exception(MessageHeaderSuppressor, Messages&&... messages) noexcept
		    : Exception(std::forward<Messages>(messages)...)
		{
			/* NO CODE */
		}


	private:
		template <typename... Messages>
		void BuildMessage(Messages&&... messages) noexcept
		{
			if (std::uncaught_exceptions() > 0)
			{
				m_CMessage = "Exception thrown during stack unwinding";
				return;
			}

			try
			{
				const std::size_t totalSize = (MessageSize(messages) + ...)
				                              + JoinerSize() * (sizeof...(messages) > 0 ? sizeof...(messages) - 1 : 0);

				m_Message.reserve(totalSize);

				AppendAll(std::forward<Messages>(messages)...);

				m_CMessage = m_Message.c_str();
			}
			catch (...)
			{
				m_CMessage = "Exception message construction failed";
			}
		}

		static constexpr std::size_t JoinerSize()
		{
			return 5;  // length of "\n\t-> "
		}

		template <typename T>
		static std::size_t MessageSize(const T& msg)
		{
			if constexpr (std::is_convertible_v<T, std::string_view>)
			{
				return std::string_view(msg).size();
			}
			else
			{
				return std::to_string(msg).size();
			}
		}

		template <typename T>
		void AppendOne(T&& msg)
		{
			if constexpr (std::is_convertible_v<T, std::string_view>)
			{
				m_Message.append(std::string_view(msg));
			}
			else
			{
				m_Message.append(std::to_string(msg));
			}
		}

		template <typename First, typename... Rest>
		void AppendAll(First&& first, Rest&&... rest)
		{
			AppendOne(std::forward<First>(first));

			if constexpr (sizeof...(Rest) > 0)
			{
				m_Message.append(Joiner);
				AppendAll(std::forward<Rest>(rest)...);
			}
		}

	private:
		std::string m_Message{};
		const char* m_CMessage = "Generic exception";
	};


#define SEC_DEFINE_DERIVED_EXCEPTION_TYPE(DERIVED_EXCEPTION, BASE_EXCEPTION_TYPE)                                      \
	class DERIVED_EXCEPTION : public BASE_EXCEPTION_TYPE                                                               \
	{                                                                                                                  \
		using Base = BASE_EXCEPTION_TYPE;                                                                              \
                                                                                                                       \
	public:                                                                                                            \
		DERIVED_EXCEPTION() noexcept : Base(Exception::MessageHeaderSuppressor{}, __func__)                            \
		{                                                                                                              \
			/* NO CODE */                                                                                              \
		}                                                                                                              \
                                                                                                                       \
		template <typename... Messages>                                                                                \
		explicit DERIVED_EXCEPTION(Messages&&... messages) noexcept                                                    \
		    : Base(Exception::MessageHeaderSuppressor{}, __func__, std::forward<Messages>(messages)...)                \
		{                                                                                                              \
			/* NO CODE */                                                                                              \
		}                                                                                                              \
                                                                                                                       \
	protected:                                                                                                         \
		template <typename... Messages>                                                                                \
		explicit DERIVED_EXCEPTION(Exception::MessageHeaderSuppressor, Messages&&... messages) noexcept                \
		    : Base(Exception::MessageHeaderSuppressor{}, std::forward<Messages>(messages)...)                          \
		{                                                                                                              \
			/* NO CODE */                                                                                              \
		}                                                                                                              \
	};

	// clang-format off
	SEC_DEFINE_DERIVED_EXCEPTION_TYPE(LogicException, Exception)
		SEC_DEFINE_DERIVED_EXCEPTION_TYPE(InvalidArgumentException, LogicException)
			SEC_DEFINE_DERIVED_EXCEPTION_TYPE(ArgumentNullException, InvalidArgumentException)
			SEC_DEFINE_DERIVED_EXCEPTION_TYPE(ArgumentOutOfRangeException, InvalidArgumentException)
		SEC_DEFINE_DERIVED_EXCEPTION_TYPE(InvalidOperationException, LogicException)
		SEC_DEFINE_DERIVED_EXCEPTION_TYPE(NotImplementedException, LogicException)
		SEC_DEFINE_DERIVED_EXCEPTION_TYPE(NotSupportedException, LogicException)
		SEC_DEFINE_DERIVED_EXCEPTION_TYPE(PreconditionViolationException, LogicException)
		SEC_DEFINE_DERIVED_EXCEPTION_TYPE(PostconditionViolationException, LogicException)
		SEC_DEFINE_DERIVED_EXCEPTION_TYPE(InvariantViolationException, LogicException)
	SEC_DEFINE_DERIVED_EXCEPTION_TYPE(RuntimeException, Exception)
		SEC_DEFINE_DERIVED_EXCEPTION_TYPE(IOException, RuntimeException)
		SEC_DEFINE_DERIVED_EXCEPTION_TYPE(TimeoutException, RuntimeException)
		SEC_DEFINE_DERIVED_EXCEPTION_TYPE(OperationCanceledException, RuntimeException)
		SEC_DEFINE_DERIVED_EXCEPTION_TYPE(ResourceException, RuntimeException)
	// clang-format on

#undef SEC_DEFINE_DERIVED_EXCEPTION_TYPE


	class UnreachableException final : public LogicException
	{
		using Base = LogicException;

	public:
		UnreachableException() noexcept : Base(MessageHeaderSuppressor{}, __func__)
		{
			std::puts(__func__);
#if !(defined(SEC_ALLOW_UNREACHABLE_EXCEPTION_TO_BE_CAUGHT) && SEC_ALLOW_UNREACHABLE_EXCEPTION_TO_BE_CAUGHT)
			std::puts("Note: To make UnreachableException handleable, please recompile with ");
			std::puts("SEC_ALLOW_UNREACHABLE_EXCEPTION_TO_BE_CAUGHT=true");
			std::terminate();
#endif
		}

		template <typename... Messages>
		explicit UnreachableException(Messages&&... messages) noexcept
		    : Base(MessageHeaderSuppressor{}, __func__, std::forward<Messages>(messages)...)
		{
			std::puts(Base::what());
#if !(defined(SEC_ALLOW_UNREACHABLE_EXCEPTION_TO_BE_CAUGHT) && SEC_ALLOW_UNREACHABLE_EXCEPTION_TO_BE_CAUGHT)
			std::puts("Note: To make UnreachableException handleable, please recompile with");
			std::puts("SEC_ALLOW_UNREACHABLE_EXCEPTION_TO_BE_CAUGHT=true");
			std::terminate();
#endif
		}
	};
}
