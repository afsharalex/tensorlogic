#pragma once

#include "TL/AST.hpp"
#include <string>
#include <string_view>
#include <stdexcept>

namespace tl {

struct ParseError final : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Parse a program from a string (full file content)
Program parseProgram(std::string_view source);

// Convenience: parse a file from disk
Program parseFile(const std::string& path);

} // namespace tl
