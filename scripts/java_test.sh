#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${ROOT_DIR}/build/java-tests"
NATIVE_BUILD_DIR="${BUILD_DIR}/native"
GRADLE_PROJECT_DIR="${ROOT_DIR}/java"

for cmd in cmake java javac; do
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "error: required command '${cmd}' not found in PATH" >&2
        exit 1
    fi
done

mkdir -p "${NATIVE_BUILD_DIR}"

GRADLEW_PATH="${GRADLE_PROJECT_DIR}/gradlew"
if [[ -x "${GRADLEW_PATH}" ]]; then
    GRADLE_CMD=("${GRADLEW_PATH}")
elif command -v gradle >/dev/null 2>&1; then
    GRADLE_CMD=("gradle")
else
    echo "error: Gradle executable not found. Install Gradle or provide a Gradle wrapper in ${GRADLE_PROJECT_DIR}." >&2
    exit 1
fi

cmake -S "${ROOT_DIR}" -B "${NATIVE_BUILD_DIR}" -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF >/dev/null
cmake --build "${NATIVE_BUILD_DIR}" --target matrix_api matrix_jni >/dev/null

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        separator=';'
        ;;
    *)
        separator=':'
        ;;
esac

library_path="${NATIVE_BUILD_DIR}/native${separator}${NATIVE_BUILD_DIR}/jni"

(cd "${GRADLE_PROJECT_DIR}" && "${GRADLE_CMD[@]}" test --no-daemon --console=plain "-PnativeLibraryPath=${library_path}")