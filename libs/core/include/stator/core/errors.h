#pragma once

#include <stdexcept>
#include <format>

namespace stator::core {

//==============================================================================
// Top Level Error
//==============================================================================
class StatorError : public std::invalid_argument
{
public:
    using std::invalid_argument::invalid_argument;

    // Forward std::format() like calls to std::format()
    template <typename... Args>
    explicit StatorError(std::format_string<Args...> fmt, Args&&... args)
        : std::invalid_argument(std::format(fmt, std::forward<Args>(args)...))
    {
    }
};


//==============================================================================
// Numerics Error
//==============================================================================
class NumericError : public StatorError
{
public:
    using StatorError::StatorError;
};

class ConvergenceFailure : public NumericError
{
public:
    using NumericError::NumericError;
};

class InvalidArgument : public NumericError
{
public:
    using NumericError::NumericError;
};



}
