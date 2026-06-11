// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

namespace SecUtility
{
	template <class... Ts>
	struct OverloadSet : Ts...
	{
		using Ts::operator()...;
	};

	template <class... Ts>
	OverloadSet(Ts...) -> OverloadSet<Ts...>;
}
