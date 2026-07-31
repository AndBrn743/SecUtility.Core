# PatchIfNeeded.cmake
#
# Idempotently applies a patch file via `patch`. Runs a --dry-run first; if the
# dry-run succeeds (patch needs to be applied), applies it for real. Real apply
# failures abort with FATAL_ERROR.
#
# A non-zero dry-run exit is ambiguous and must be disambiguated by inspecting
# the output:
#   * "already applied" / "Reversed" / "hunks ignored"  -> silently skip
#   * "FAILED" / "malformed" / "patch: " / etc.         -> FATAL_ERROR
#
# The second branch exists because a stale source tree plus a broken patch can
# produce a non-zero dry-run that *looks* like "already applied" if you only
# check the exit code. Silently masking that leaves the source unpatched and
# breaks the build downstream with a confusing error far from the cause.
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
    # Non-zero dry-run exit. Scan the combined output for patch's well-known
    # failure markers; if any are present this is a genuine patch failure (not
    # "already applied") and must abort loudly. Otherwise the patch was already
    # applied and we skip silently.
    set(_dry_run_log "${_dry_run_out}${_dry_run_err}")
    string(REGEX MATCH
        "FAILED|malformed|unexpectedly|patch: |No file to patch"
        _failure_match "${_dry_run_log}")
    if(_failure_match)
        message(FATAL_ERROR
            "Patch does not apply cleanly: ${PATCH_FILE}\n"
            "  source dir:    ${SOURCE_DIR}\n"
            "  exit code:     ${_dry_run_result}\n"
            "  dry-run stdout: ${_dry_run_out}\n"
            "  dry-run stderr: ${_dry_run_err}")
    endif()
    message(STATUS "Patch already applied or unnecessary: ${PATCH_FILE}")
endif()
