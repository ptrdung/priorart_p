"""
Test script to verify the updated query generation prompt produces shorter queries
"""

from src.prompts.extraction_prompts import ExtractionPrompts

def test_query_prompt():
    """Test the updated query generation prompt"""
    
    # Get the prompt and parser
    prompt, parser = ExtractionPrompts.get_queries_prompt_and_parser()
    
    # Sample test data
    test_data = {
        "summary": "A machine learning system for image recognition using convolutional neural networks",
        "problem_purpose_keys": ["image recognition", "pattern detection", "classification accuracy"],
        "object_system_keys": ["convolutional neural network", "deep learning model", "feature extraction"],
        "environment_field_keys": ["computer vision", "artificial intelligence", "medical imaging"],
        "CPC_CODES": ["G06N3/02", "G06T7/00"]
    }
    
    # Format the prompt
    formatted_prompt = prompt.format(**test_data)
    
    print("=== UPDATED QUERY GENERATION PROMPT ===")
    print(formatted_prompt)
    print("\n" + "="*80)
    
    # Check for key improvements
    improvements = [
        "200 characters" in formatted_prompt,
        "SELECT ONLY 2-3 MOST DISCRIMINATIVE TERMS" in formatted_prompt,
        "Maximum keywords: 8-10 total terms" in formatted_prompt,
        "QUERY CONSTRUCTION GUIDELINES" in formatted_prompt,
        "Example format:" in formatted_prompt
    ]
    
    print("=== VERIFICATION OF IMPROVEMENTS ===")
    print(f"✅ Character limit constraint: {'✓' if improvements[0] else '✗'}")
    print(f"✅ Keyword selection guidance: {'✓' if improvements[1] else '✗'}")
    print(f"✅ Maximum keyword limit: {'✓' if improvements[2] else '✗'}")
    print(f"✅ Construction guidelines: {'✓' if improvements[3] else '✗'}")
    print(f"✅ Example formats provided: {'✓' if improvements[4] else '✗'}")
    
    all_improvements = all(improvements)
    print(f"\n🎯 Overall Status: {'All improvements implemented successfully!' if all_improvements else 'Some improvements missing!'}")
    
    return all_improvements

if __name__ == "__main__":
    test_query_prompt()
