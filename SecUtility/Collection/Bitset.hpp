// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Andy Brown

#pragma once


#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winvalid-constexpr"
#endif

#define SEC_BITSET_DETAIL
// clang-format off
#include <SecUtility/Collection/Detail/Bitset.Forward.hpp>
#include <SecUtility/Collection/Detail/Bitset.Utility.hpp>
#include <SecUtility/Collection/Detail/Bitset.Base.hpp>
#include <SecUtility/Collection/Detail/Bitset.Impl.hpp>
// clang-format on
#undef SEC_BITSET_DETAIL


#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
