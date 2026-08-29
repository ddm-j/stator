#pragma once

#include <stdexcept>

namespace stator::core {

//==============================================================================
// Top Level Error
//==============================================================================
class StatorError : public std::invalid_argument
{
public:
    using std::invalid_argument::invalid_argument;
};


//==============================================================================
// Numerics Error
//==============================================================================
class NumericError : public StatorError
{
public:
    using StatorError::StatorError;
};

class MaxIterations : public NumericError
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
