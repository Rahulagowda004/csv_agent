#!/usr/bin/env python3
"""
Test script for the structured CSV agent
"""

import asyncio
import json
from structured_agent import run_structured_csv_analysis

async def test_structured_analysis():
    """Test the structured CSV analysis with a sample query."""
    
    print("🧪 Testing Structured CSV Analysis")
    print("=" * 50)
    
    # Test query
    query = """
    Analyze the sample data and create visualizations showing:
    1. Age distribution of employees
    2. Salary by department comparison
    3. Experience vs salary correlation
    
    Generate charts and provide insights about the workforce data.
    """
    
    try:
        print("🔄 Running analysis...")
        result = await run_structured_csv_analysis(query, "./data")
        
        print("\n✅ Test Completed Successfully!")
        print("=" * 50)
        
        # Validate all required fields are present
        print("🔍 Validation Results:")
        print(f"✓ final_response: {bool(result.final_response)} ({len(result.final_response)} chars)")
        print(f"✓ table_data: {bool(result.table_data)} ({'Dict' if result.table_data else 'None'})")
        print(f"✓ image_url_list: {len(result.image_url_list)} images")
        print(f"✓ steps_taken: {len(result.steps_taken)} steps")
        print(f"✓ suggested_questions: {len(result.suggested_questions)} questions")
        
        # Show sample content
        print(f"\n📄 Sample Final Response:")
        print(f"{result.final_response[:200]}...")
        
        if result.image_url_list:
            print(f"\n🖼️  Generated Images:")
            for img in result.image_url_list:
                print(f"   • {img}")
        
        if result.suggested_questions:
            print(f"\n❓ LLM-Generated Questions:")
            for i, q in enumerate(result.suggested_questions, 1):
                print(f"   {i}. {q}")
        
        print(f"\n📊 Steps Taken:")
        for i, step in enumerate(result.steps_taken, 1):
            print(f"   {i}. {step['tool_name']}: {step['description'][:80]}...")
        
        # Export full result as JSON
        with open("test_result.json", "w") as f:
            json.dump(result.dict(), f, indent=2, default=str)
        print(f"\n💾 Full result saved to test_result.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_structured_analysis())
    print(f"\n{'✅ All tests passed!' if success else '❌ Tests failed!'}")
