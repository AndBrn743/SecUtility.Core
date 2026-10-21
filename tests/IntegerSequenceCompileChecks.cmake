# Invalid ranges are hard compilation errors. Compile each case independently;
# positive controls ensure a broken toolchain or missing header cannot pass them.
get_filename_component(integer_sequence_include_dir "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${integer_sequence_include_dir}/SecUtility/Meta/IntegerSequence.hpp"
        "${integer_sequence_include_dir}/SecUtility/Meta/StaticForeach.hpp")

function(check_integer_sequence_compilation name expression expected)
    set(check_dir "${CMAKE_CURRENT_BINARY_DIR}/integer-sequence-compile-checks")
    file(MAKE_DIRECTORY "${check_dir}")
    set(source "${check_dir}/${name}.cpp")
    file(WRITE "${source}"
            "#include <SecUtility/Meta/StaticForeach.hpp>\n"
            "#include <cstdint>\n#include <limits>\n#include <type_traits>\n"
            "struct Operation { void operator()(int) const {} };\n"
            "using Result = ${expression};\n"
            "static_assert(std::is_same_v<Result, Result>);\n")
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    try_compile(compiled "${check_dir}/${name}" "${source}"
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${integer_sequence_include_dir}"
            CXX_STANDARD ${CMAKE_CXX_STANDARD}
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
            OUTPUT_VARIABLE output)
    file(WRITE "${check_dir}/${name}.log" "${output}")
    if (expected AND NOT compiled)
        message(FATAL_ERROR "Integer sequence positive control ${name} failed:\n${output}")
    elseif (NOT expected AND compiled)
        message(FATAL_ERROR "Invalid integer sequence ${name} unexpectedly compiled")
    endif ()
endfunction()

check_integer_sequence_compilation(empty "SecUtility::MakeIndexSequence<0>" TRUE)
check_integer_sequence_compilation(ascending "SecUtility::MakeIntegerSequence<int, -3, 4, 2>" TRUE)
check_integer_sequence_compilation(descending "SecUtility::MakeIntegerSequence<unsigned, 5, 1, -2>" TRUE)
check_integer_sequence_compilation(foreach "decltype(SecUtility::StaticForeachInRange<3>(Operation{}))" TRUE)

check_integer_sequence_compilation(no_bounds "SecUtility::MakeIntegerSequence<int>" FALSE)
check_integer_sequence_compilation(too_many_bounds "SecUtility::MakeIntegerSequence<int, 0, 1, 1, 1>" FALSE)
check_integer_sequence_compilation(nonintegral_index "SecUtility::MakeIntegerSequence<double, 3>" FALSE)
check_integer_sequence_compilation(nonintegral_bound "SecUtility::MakeIntegerSequence<int, nullptr>" FALSE)
check_integer_sequence_compilation(nonintegral_step "SecUtility::MakeIntegerSequence<int, 0, 3, nullptr>" FALSE)
check_integer_sequence_compilation(negative_count "SecUtility::MakeIntegerSequence<int, -1>" FALSE)
check_integer_sequence_compilation(zero_step "SecUtility::MakeIntegerSequence<int, 0, 3, 0>" FALSE)
check_integer_sequence_compilation(empty_zero_step "SecUtility::MakeIntegerSequence<int, 3, 3, 0>" FALSE)
check_integer_sequence_compilation(reversed_bounds "SecUtility::MakeIntegerSequence<int, 3, 1>" FALSE)
check_integer_sequence_compilation(wrong_negative_step "SecUtility::MakeIntegerSequence<int, 1, 3, -1>" FALSE)
check_integer_sequence_compilation(wrong_positive_step "SecUtility::MakeIntegerSequence<int, 3, 1, 1>" FALSE)
check_integer_sequence_compilation(negative_unsigned_start "SecUtility::MakeIntegerSequence<unsigned, -1, 3>" FALSE)
check_integer_sequence_compilation(negative_unsigned_end "SecUtility::MakeIntegerSequence<unsigned, 3, -1, -1>" FALSE)
check_integer_sequence_compilation(narrow_count "SecUtility::MakeIntegerSequence<unsigned char, 256>" FALSE)
check_integer_sequence_compilation(narrow_end "SecUtility::MakeIntegerSequence<unsigned char, 0, 256>" FALSE)
check_integer_sequence_compilation(narrow_start "SecUtility::MakeIntegerSequence<unsigned char, 256, 0, -1>" FALSE)
check_integer_sequence_compilation(narrow_bool "SecUtility::MakeIntegerSequence<bool, 2>" FALSE)
check_integer_sequence_compilation(negative_index_count "SecUtility::MakeIndexSequence<-1>" FALSE)
check_integer_sequence_compilation(negative_index_start "SecUtility::MakeIndexSequence<-1, 3>" FALSE)
check_integer_sequence_compilation(negative_index_end "SecUtility::MakeIndexSequence<3, -1, -1>" FALSE)
check_integer_sequence_compilation(narrow_signed_end
        "SecUtility::MakeIntegerSequence<std::intmax_t, 0, std::numeric_limits<std::uintmax_t>::max()>" FALSE)

# Range wrappers must also propagate invalid range diagnostics to their callers.
check_integer_sequence_compilation(foreach_negative_count
        "decltype(SecUtility::StaticForeachInRange<-1>(Operation{}))" FALSE)
check_integer_sequence_compilation(foreach_reversed_bounds
        "decltype(SecUtility::StaticForeachInRange<3, 1>(Operation{}))" FALSE)
check_integer_sequence_compilation(foreach_zero_step
        "decltype(SecUtility::StaticForeachInRange<0, 3, 0>(Operation{}))" FALSE)
check_integer_sequence_compilation(foreach_wrong_direction
        "decltype(SecUtility::StaticForeachInRange<0, 3, -1>(Operation{}))" FALSE)
check_integer_sequence_compilation(foreach_mixed_bounds
        "decltype(SecUtility::StaticForeachInRange<-3, 4u>(Operation{}))" FALSE)
check_integer_sequence_compilation(foreach_mixed_stepped_bounds
        "decltype(SecUtility::StaticForeachInRange<-3, 4u, 2>(Operation{}))" FALSE)
