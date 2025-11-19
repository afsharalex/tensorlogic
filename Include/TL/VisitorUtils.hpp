// Include/TL/VisitorUtils.hpp
// Utilities for std::visit pattern matching

#pragma once

namespace tl {

// Overloaded lambda helper for std::visit
// Usage:
//   std::visit(overloaded{
//       [](const Type1& x) { ... },
//       [](const Type2& x) { ... },
//       ...
//   }, variant);
template<class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

// C++17 deduction guide
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

} // namespace tl
