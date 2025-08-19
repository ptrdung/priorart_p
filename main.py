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
        
        if results["concept_matrix"]:
            print("\n🎯 Concept Matrix:")
            for key, value in results["concept_matrix"].items():
                print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        if results["seed_keywords"]:
            print("\n🔑 Seed Keywords:")
            for key, keywords in results["seed_keywords"].dict().items():
                print(f"  • {key.replace('_', ' ').title()}: {keywords}")
        
        if results["final_keywords"]:
            print("\n🎨 Enhanced Keywords:")
            for key, synonyms in results["final_keywords"].items():
                print(f"  • {key}: {synonyms}")
        
        if results["queries"]:
            print("\n🔍 Generated Queries:")
            for i, query in enumerate(results["queries"].queries, 1):
                print(f"  {i}. {query}")
        
        if results["final_url"]:
            print(f"\n🌐 Found {len(results['final_url'])} relevant patents")
            for i, url_data in enumerate(results["final_url"][:3], 1):
                print(f"  {i}. {url_data.get('url', 'N/A')} (Score: {url_data.get('user_scenario', 0):.2f})")
        
        print(f"\n🎬 User Action: {results.get('user_action', 'N/A')}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
