#pragma once

namespace slam {

enum class ExecutionMode { Serial, SerialSimd, Parallel };

// Backend compiled for ExecutionMode::SerialSimd: "neon", "sse2", "vector-extensions"
// (portable GCC/Clang vectors) or "scalar" (no vector support in the compiler).
const char *simdBackendName() noexcept;

} // namespace slam
