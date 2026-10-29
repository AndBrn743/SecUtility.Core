// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2026 Andy Brown

#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define SEC_FLATTEN __attribute__((flatten))
#else
#define SEC_FLATTEN
#endif
