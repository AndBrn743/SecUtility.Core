# PatchIfNeeded.cmake
#
# Idempotently applies a patch file via `patch`. Runs a --dry-run first; if the
# dry-run succeeds (patch needs to be applied), applies it for real. If the
# dry-run fails (patch already applied, reversed, or unnecessary), exits cleanly
# and reports the dry-run stderr so the cause is visible in the configure log.
# Real apply failures abort with FATAL_ERROR so they aren't silently masked.
#
# Required arguments (all of):
#   -DPATCH_EXE=<path>     Path to the patch executable.
#   -DPATCH_FILE=<path>    Path to the .patch file to apply.
#   -DSOURCE_DIR=<dir>     Working directory in which `patch` runs.
#   -DSTRIP=<n>            Strip level (the -p<n> argument).
#
# Uses `-i <file>` instead of stdin redirection so no shell is required; this
# keeps the wrapper usable on platforms without a POSIX shell (Windows).

if(NOT DEFINED PATCH_EXE OR NOT DEFINED PATCH_FILE OR NOT DEFINED SOURCE_DIR OR NOT DEFINED STRIP)
    message(FATAL_ERROR
        "PatchIfNeeded.cmake requires -DPATCH_EXE, -DPATCH_FILE, -DSOURCE_DIR, -DSTRIP")
endif()

execute_process(
    COMMAND ${PATCH_EXE} --forward --batch --silent --dry-run -p${STRIP} -i ${PATCH_FILE}
    WORKING_DIRECTORY ${SOURCE_DIR}
    RESULT_VARIABLE _dry_run_result
    OUTPUT_VARIABLE _dry_run_out
    ERROR_VARIABLE _dry_run_err)

if(_dry_run_result EQUAL 0)
    message(STATUS "Applying patch: ${PATCH_FILE}")
    execute_process(
        COMMAND ${PATCH_EXE} --forward --batch --silent -p${STRIP} -i ${PATCH_FILE}
        WORKING_DIRECTORY ${SOURCE_DIR}
        RESULT_VARIABLE _apply_result
        OUTPUT_VARIABLE _apply_out
        ERROR_VARIABLE _apply_err)
    if(NOT _apply_result EQUAL 0)
        message(FATAL_ERROR
            "Patch failed: ${PATCH_FILE}\n"
            "  exit code: ${_apply_result}\n"
            "  stderr: ${_apply_err}")
    endif()
else()
    message(STATUS "Patch already applied or unnecessary: ${PATCH_FILE}")
    if(_dry_run_err)
        message(STATUS "  dry-run stderr: ${_dry_run_err}")
    endif()
endif()
