// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#if defined(_MSC_VER)
#define SEC_FORCE_INLINE __forceinline
#else
#define SEC_FORCE_INLINE __attribute__((always_inline)) inline
#endif
