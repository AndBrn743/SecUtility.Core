//
// Created by Andy on 5/6/2026.
//

#pragma once

#include <ostream>

namespace SecUtility
{
	inline int BitOrderFlag()
	{
		static const int index = std::ios_base::xalloc();
		return index;
	}

	inline std::ostream& MsbFirst(std::ostream& os)
	{
		os.iword(BitOrderFlag()) = 0;
		return os;
	}

	inline std::ostream& LsbFirst(std::ostream& os)
	{
		os.iword(BitOrderFlag()) = 1;
		return os;
	}
}
