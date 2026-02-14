#!/bin/bash
# Helper script to run GitHub Actions workflows locally using act
# This script uses system compilers instead of Docker containers

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if act is installed
if ! command -v act &> /dev/null; then
    echo_error "act is not installed. Install it from https://github.com/nektos/act"
    echo_info "On Arch: yay -S act"
    exit 1
fi

# Check for required system packages
echo_info "Checking for required system packages..."

MISSING_PACKAGES=()

check_package() {
    if ! command -v "$1" &> /dev/null; then
        MISSING_PACKAGES+=("$2")
    fi
}

check_package cmake cmake
check_package ninja ninja
check_package gcc gcc
check_package clang clang
check_package g++ gcc
check_package clang++ clang
check_package lcov lcov
check_package ccache ccache

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo_error "Missing required packages: ${MISSING_PACKAGES[*]}"
    echo_info "Install them with: sudo pacman -S ${MISSING_PACKAGES[*]}"
    exit 1
fi

echo_info "All required packages are installed"

# Check for cached Catch2 tarball
CATCH2_TARBALL="$PROJECT_ROOT/.cache/catch2-v3.7.1.tar.gz"
if [ -f "$CATCH2_TARBALL" ]; then
    echo_info "Using cached Catch2: $CATCH2_TARBALL"
    export CATCH2_SOURCE="$CATCH2_TARBALL"
else
    echo_warn "No cached Catch2 found at $CATCH2_TARBALL"
    echo_warn "Run: mkdir -p .cache && cd .cache && curl -L -o catch2-v3.7.1.tar.gz https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz"
fi

# Check for cached nlohmann/json tarball
NLOHMANN_JSON_TARBALL="$PROJECT_ROOT/.cache/nlohmann_json-v3.12.0.tar.xz"
if [ -f "$NLOHMANN_JSON_TARBALL" ]; then
    echo_info "Using cached nlohmann/json tarball: $NLOHMANN_JSON_TARBALL"
    export NLOHMANN_JSON_SOURCE="$NLOHMANN_JSON_TARBALL"
else
    echo_warn "No cached nlohmann/json tarball found at $NLOHMANN_JSON_TARBALL"
    echo_warn "Run: mkdir -p .cache && cd .cache && curl -L -o nlohmann_json-v3.12.0.tar.xz https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz"
fi

# Parse arguments
JOB=""
WORKFLOW=".github/workflows/local.yml"

while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--job)
            JOB="$2"
            shift 2
            ;;
        -w|--workflow)
            WORKFLOW="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -j, --job JOB     Run specific job (local-gcc, local-clang-libstdcxx, local-clang-libcxx, local-coverage)"
            echo "  -w, --workflow    Use specific workflow file (default: .github/workflows/local.yml)"
            echo "  --no-cache        Disable using cached Catch2 and nlohmann/json (force download)"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all local jobs"
            echo "  $0 -j local-gcc                       # Run GCC job only"
            echo "  $0 -j local-clang-libcxx              # Run Clang with libc++ job"
            echo "  $0 -j local-coverage                  # Run coverage job"
            echo "  $0 --no-cache -j local-gcc            # Run without using cached Catch2 and nlohmann/json"
            exit 0
            ;;
        --no-cache)
            unset CATCH2_SOURCE
            echo_info "Catch2 cache disabled"
            shift
            ;;
        *)
            echo_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build act command
ACT_CMD="act -W $WORKFLOW"

# Use system act without Docker by using the host network
# This allows access to system-installed compilers
ACT_CMD="$ACT_CMD -P self-hosted=-self-hosted"

# Pass CATCH2_SOURCE as environment variable if set
if [ -n "$CATCH2_SOURCE" ]; then
    ACT_CMD="$ACT_CMD --env CATCH2_SOURCE=$CATCH2_SOURCE"
    echo_info "Using local Catch2 cache"
fi

if [ -n "$JOB" ]; then
    echo_info "Running job: $JOB"
    ACT_CMD="$ACT_CMD -j $JOB"
else
    echo_info "Running all local jobs"
fi

echo_info "Executing: $ACT_CMD"
echo ""

# Execute act
eval $ACT_CMD
