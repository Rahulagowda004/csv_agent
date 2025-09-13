#!/usr/bin/env python3
"""
Test script to verify table display functionality.
"""

import sys
import os
sys.path.append('frontend/src')

# Mock streamlit for testing
class MockStreamlit:
    def dataframe(self, data, use_container_width=True):
        print(f"📊 Displaying DataFrame with {len(data)} rows and {len(data.columns)} columns")
        print(f"Columns: {list(data.columns)}")
        print(f"First few rows:\n{data.head()}")
    
    def subheader(self, text):
        print(f"📋 {text}")
    
    def write(self, text):
        print(f"📝 {text}")
    
    def expander(self, text):
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def json(self, data):
        print(f"🔧 JSON fallback: {type(data)}")

# Mock streamlit module
sys.modules['streamlit'] = MockStreamlit()

import pandas as pd
from chat_interface import ChatInterface

def test_table_display():
    """Test the table display functionality with various data structures."""
    
    print("🧪 Testing table display functionality...")
    
    # Create a mock API client
    class MockAPIClient:
        pass
    
    # Create chat interface instance
    chat_interface = ChatInterface(MockAPIClient())
    
    # Test 1: Simple DataFrame-like structure
    print("\n📊 Test 1: DataFrame-like structure")
    table_data_1 = {
        'data': [
            [1, 'Alice', 25],
            [2, 'Bob', 30],
            [3, 'Charlie', 35]
        ],
        'columns': ['ID', 'Name', 'Age']
    }
    chat_interface.display_table(table_data_1)
    
    # Test 2: Sample data structure (like from CSV analysis)
    print("\n📊 Test 2: Sample data structure")
    table_data_2 = {
        'sample_data': {
            'first_5_rows': [
                {'product': 'Widget A', 'price': 10.99, 'quantity': 100},
                {'product': 'Widget B', 'price': 15.99, 'quantity': 50},
                {'product': 'Widget C', 'price': 8.99, 'quantity': 200}
            ]
        }
    }
    chat_interface.display_table(table_data_2)
    
    # Test 3: Statistics data
    print("\n📊 Test 3: Statistics data")
    table_data_3 = {
        'numeric_statistics': {
            'price': {'mean': 12.99, 'std': 3.5, 'min': 8.99, 'max': 15.99},
            'quantity': {'mean': 116.67, 'std': 76.38, 'min': 50, 'max': 200}
        }
    }
    chat_interface.display_table(table_data_3)
    
    # Test 4: Simple key-value structure
    print("\n📊 Test 4: Key-value structure")
    table_data_4 = {
        'total_rows': 1000,
        'total_columns': 5,
        'missing_values': 25,
        'duplicate_rows': 10
    }
    chat_interface.display_table(table_data_4)
    
    print("\n✅ All table display tests completed!")

if __name__ == "__main__":
    test_table_display()
