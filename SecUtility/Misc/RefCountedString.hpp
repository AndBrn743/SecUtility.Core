// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#include <cstddef>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>

namespace SecUtility
{
	class RefCountedString
	{
	public:
		RefCountedString() : _impl(std::make_shared<const std::string>())
		{
			/* NO CODE */
		}

		/* IMPLICIT */ RefCountedString(std::string s) : _impl(std::make_shared<const std::string>(std::move(s)))
		{
			/* NO CODE */
		}

		/* IMPLICIT */ RefCountedString(std::string_view sv) : _impl(std::make_shared<const std::string>(sv))
		{
			/* NO CODE */
		}

		/* IMPLICIT */ RefCountedString(const char* s) : _impl(std::make_shared<const std::string>(s))
		{
			/* NO CODE */
		}

		RefCountedString(const RefCountedString&) noexcept = default;
		RefCountedString& operator=(const RefCountedString&) noexcept = default;
		RefCountedString(RefCountedString&&) noexcept = default;
		RefCountedString& operator=(RefCountedString&&) noexcept = default;
		~RefCountedString() = default;

		const char* CString() const noexcept
		{
			return _impl->c_str();
		}

		std::size_t Size() const noexcept
		{
			return _impl->size();
		}

		std::size_t Length() const noexcept
		{
			return _impl->length();
		}

		char operator[](const std::size_t index) const noexcept
		{
			return (*_impl)[index];
		}

		friend std::ostream& operator<<(std::ostream& os, const RefCountedString& str)
		{
			return os << *str._impl;
		}

	private:
		std::shared_ptr<const std::string> _impl;
	};
}
