// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define SEC_LIKELY(EXPR) __builtin_expect(static_cast<bool>(EXPR), true)
#else
#define SEC_LIKELY(EXPR) static_cast<bool>(EXPR)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SEC_UNLIKELY(EXPR) __builtin_expect(static_cast<bool>(EXPR), false)
#else
#define SEC_UNLIKELY(EXPR) static_cast<bool>(EXPR)
#endif
