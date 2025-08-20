"""
Main entry point for the Patent AI Agent
Provides a simple interface to run the patent keyword extraction system
"""

from src.core.extractor import CoreConceptExtractor

def main():
    """Main function to run the patent keyword extraction"""
    
    # Sample patent idea text for testing
    sample_text = """
    **Idea title**: Smart Irrigation System with IoT Sensors

    **User scenario**: A farmer managing a large agricultural field needs to optimize water usage 
    while ensuring crops receive adequate moisture. The farmer wants to monitor soil conditions 
    remotely and automatically adjust irrigation based on real-time data from multiple field locations.

    **User problem**: Traditional irrigation systems either over-water or under-water crops because 
    they operate on fixed schedules without considering actual soil moisture, weather conditions, 
    or crop-specific needs. This leads to water waste, increased costs, and potentially reduced 
    crop yields.
    """
    
    print("🚀 Starting Patent AI Agent - Keyword Extraction System")
    print("=" * 60)
    
    # Initialize the extractor
    extractor = CoreConceptExtractor()
    
    # Run the extraction workflow
    print("\n📝 Processing patent idea...")
    print(f"Input text: {sample_text[:100]}...")
    
    try:
        results = extractor.extract_keywords(sample_text)
        
        print("\n✅ Extraction completed!")
        print("\n📊 Results Summary:")
        print("-" * 40)

        # Print all ExtractionState fields for full transparency
        for key, value in results.items():
            if value is None:
                continue
            print(f"\n🔹 {key}:")
            if hasattr(value, "dict"):
                for subkey, subval in value.dict().items():
                    print(f"  • {subkey.replace('_', ' ').title()}: {subval}")
            elif isinstance(value, dict):
                for subkey, subval in value.items():
                    print(f"  • {subkey}: {subval}")
            elif isinstance(value, list):
                for i, item in enumerate(value, 1):
                    if isinstance(item, dict):
                        print(f"  {i}. " + ", ".join([f"{k}: {v}" for k, v in item.items()]))
                    else:
                        print(f"  {i}. {item}")
            else:
                print(f"  {value}")

        # Save results to JSON file
        import json
        import datetime
        filename = f"extraction_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            def serialize(obj):
                if hasattr(obj, "dict"):
                    return obj.dict()
                return obj
            json.dump({k: serialize(v) for k, v in results.items()}, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {filename}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
