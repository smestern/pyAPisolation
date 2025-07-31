#!/usr/bin/env python3
"""
Test script for the new Parameter class
"""

from pyAPisolation.analysis.base import Parameter

def test_parameter_creation():
    """Test basic parameter creation"""
    print("Testing basic parameter creation...")
    
    # Basic parameter
    param1 = Parameter(
        name="threshold",
        param_type=float,
        default=5.0,
        min_value=0.0,
        max_value=100.0,
        description="Detection threshold in mV"
    )
    print(f"Created parameter: {param1.name}, type: {param1.param_type}, default: {param1.default}")
    
    # Integer parameter with options
    param2 = Parameter(
        name="method",
        param_type=str,
        default="peak",
        options=["peak", "derivative", "template"],
        description="Detection method"
    )
    print(f"Created parameter: {param2.name}, options: {param2.options}")
    
    # Boolean parameter
    param3 = Parameter(
        name="enable_filter",
        param_type=bool,
        default=True,
        description="Enable filtering"
    )
    print(f"Created parameter: {param3.name}, default: {param3.default}")

def test_parameter_validation():
    """Test parameter validation"""
    print("\nTesting parameter validation...")
    
    param = Parameter(
        name="threshold",
        param_type=float,
        default=5.0,
        min_value=0.0,
        max_value=100.0
    )
    
    # Valid values
    test_values = [5.0, 0.0, 100.0, 50.5, "10.5"]  # Last one should convert
    for value in test_values:
        try:
            converted = param.validate_and_convert(value)
            print(f"Value {value} -> {converted} (valid: {param.is_valid(converted)})")
        except ValueError as e:
            print(f"Value {value} failed: {e}")
    
    # Invalid values
    invalid_values = [-1.0, 101.0, "invalid"]
    for value in invalid_values:
        try:
            converted = param.validate_and_convert(value)
            print(f"Value {value} -> {converted} (should have failed!)")
        except ValueError as e:
            print(f"Value {value} correctly failed: {e}")

def test_options_validation():
    """Test options validation"""
    print("\nTesting options validation...")
    
    param = Parameter(
        name="method",
        param_type=str,
        default="peak",
        options=["peak", "derivative", "template"]
    )
    
    # Valid options
    for option in param.options:
        print(f"Option '{option}' is valid: {param.is_valid(option)}")
    
    # Invalid option
    try:
        param.validate_and_convert("invalid_method")
        print("Invalid method should have failed!")
    except ValueError as e:
        print(f"Invalid method correctly failed: {e}")

if __name__ == "__main__":
    test_parameter_creation()
    test_parameter_validation()
    test_options_validation()
    print("\nAll tests completed!")
