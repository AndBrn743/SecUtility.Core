// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once


#if defined(_WIN32) && defined(SECUTILITY_CORE_SHARED)
#if defined(SECUTILITY_CORE_BUILDING)
#define SECUTILITY_CORE_API __declspec(dllexport)
#else
#define SECUTILITY_CORE_API __declspec(dllimport)
#endif
#else
#define SECUTILITY_CORE_API
#endif
