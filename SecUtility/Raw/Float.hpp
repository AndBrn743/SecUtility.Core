// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once


#if defined(SECUTILITY_NO_TOP_LEVEL_FLOAT_ALIAS) && SECUTILITY_NO_TOP_LEVEL_FLOAT_ALIAS
namespace SecUtility
{
#endif

	using Double = double;
	using Single = float;

#if defined(SECUTILITY_NO_TOP_LEVEL_FLOAT_ALIAS) && SECUTILITY_NO_TOP_LEVEL_FLOAT_ALIAS
}
#endif
