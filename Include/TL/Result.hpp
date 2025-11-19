// Include/TL/Result.hpp
// Result type for error handling (similar to Rust's Result<T, E>)

#pragma once

#include <variant>
#include <utility>
#include <stdexcept>
#include <string>
#include <optional>

namespace tl {

// Result<T, E> represents either success (T) or failure (E)
template<typename T, typename E>
class Result {
public:
    // Construct from success value
    static Result Ok(T value) {
        return Result(std::move(value), true);
    }

    // Construct from error value
    static Result Err(E error) {
        return Result(std::move(error), false);
    }

    // Check if result is successful
    bool isOk() const { return is_ok_; }
    bool isErr() const { return !is_ok_; }

    // Explicit boolean conversion
    explicit operator bool() const { return isOk(); }

    // Get the success value (throws if error)
    T& value() & {
        if (!is_ok_) {
            throw std::runtime_error("Attempted to access value of error Result");
        }
        return std::get<T>(data_);
    }

    const T& value() const& {
        if (!is_ok_) {
            throw std::runtime_error("Attempted to access value of error Result");
        }
        return std::get<T>(data_);
    }

    T&& value() && {
        if (!is_ok_) {
            throw std::runtime_error("Attempted to access value of error Result");
        }
        return std::move(std::get<T>(data_));
    }

    // Get the error value (throws if success)
    E& error() & {
        if (is_ok_) {
            throw std::runtime_error("Attempted to access error of success Result");
        }
        return std::get<E>(data_);
    }

    const E& error() const& {
        if (is_ok_) {
            throw std::runtime_error("Attempted to access error of success Result");
        }
        return std::get<E>(data_);
    }

    E&& error() && {
        if (is_ok_) {
            throw std::runtime_error("Attempted to access error of success Result");
        }
        return std::move(std::get<E>(data_));
    }

    // Get value or default
    T valueOr(T default_value) const& {
        return is_ok_ ? std::get<T>(data_) : std::move(default_value);
    }

    T valueOr(T default_value) && {
        return is_ok_ ? std::move(std::get<T>(data_)) : std::move(default_value);
    }

    // Map success value to another type
    template<typename F>
    auto map(F&& func) -> Result<decltype(func(std::declval<T>())), E> {
        using U = decltype(func(std::declval<T>()));
        if (is_ok_) {
            return Result<U, E>::Ok(func(std::move(std::get<T>(data_))));
        } else {
            return Result<U, E>::Err(std::move(std::get<E>(data_)));
        }
    }

    // Map error value to another type
    template<typename F>
    auto mapErr(F&& func) -> Result<T, decltype(func(std::declval<E>()))> {
        using F2 = decltype(func(std::declval<E>()));
        if (is_ok_) {
            return Result<T, F2>::Ok(std::move(std::get<T>(data_)));
        } else {
            return Result<T, F2>::Err(func(std::move(std::get<E>(data_))));
        }
    }

    // Monadic bind operation
    template<typename F>
    auto andThen(F&& func) -> decltype(func(std::declval<T>())) {
        if (is_ok_) {
            return func(std::move(std::get<T>(data_)));
        } else {
            using ResultType = decltype(func(std::declval<T>()));
            return ResultType::Err(std::move(std::get<E>(data_)));
        }
    }

private:
    Result(T value, bool) : data_(std::move(value)), is_ok_(true) {}
    Result(E error, bool) : data_(std::move(error)), is_ok_(false) {}

    std::variant<T, E> data_;
    bool is_ok_;
};

// Specialization for void success type
template<typename E>
class Result<void, E> {
public:
    static Result Ok() {
        return Result();
    }

    static Result Err(E error) {
        return Result(std::move(error));
    }

    bool isOk() const { return is_ok_; }
    bool isErr() const { return !is_ok_; }

    explicit operator bool() const { return isOk(); }

    E& error() & {
        if (is_ok_) {
            throw std::runtime_error("Attempted to access error of success Result");
        }
        return *error_;
    }

    const E& error() const& {
        if (is_ok_) {
            throw std::runtime_error("Attempted to access error of success Result");
        }
        return *error_;
    }

    E&& error() && {
        if (is_ok_) {
            throw std::runtime_error("Attempted to access error of success Result");
        }
        return std::move(*error_);
    }

private:
    Result() : is_ok_(true), error_(std::nullopt) {}
    Result(E error) : error_(std::move(error)), is_ok_(false) {}

    std::optional<E> error_;
    bool is_ok_;
};

} // namespace tl
