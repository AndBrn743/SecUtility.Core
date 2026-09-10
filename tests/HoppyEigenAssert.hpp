// SPDX-License-Identifier: MIT

#pragma once

#define HOPPY_TEST_EIGEN_ASSERT_THROWS 1

#include <stdexcept>
#include <string>

namespace Hoppy::Test
{
	class EigenAssertionFailure : public std::runtime_error
	{
	public:
		EigenAssertionFailure(const char* expression, const char* file, int line)
			: std::runtime_error(
			          std::string("Eigen assertion failed: ") + expression + " (" + file + ":"
			          + std::to_string(line) + ")")
		{}
	};
}  // namespace Hoppy::Test

#ifdef eigen_assert
#undef eigen_assert
#endif
#define eigen_assert(condition)                                                                                       \
	((condition) ? static_cast<void>(0)                                                                                \
	             : throw ::Hoppy::Test::EigenAssertionFailure(#condition, __FILE__, __LINE__))
