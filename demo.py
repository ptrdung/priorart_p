"""
Demo script for patent seed keyword extraction system
"""

from core_concept_extractor import CoreConceptExtractor
import json


def demo_with_sample_patents():
    """Demo with different patent samples"""
    
    # Use simple mode without checkpointer to avoid thread_id requirements
    extractor = CoreConceptExtractor(model_name="qwen2.5:0.5b-instruct", use_checkpointer=False)

    samples = [
        {
            "title": "Smart Irrigation System",
            "text": """
            A smart irrigation system that uses soil moisture sensors and weather data 
            to automatically control irrigation schedules. The system includes IoT sensors, 
            a central controller, and a mobile application. It helps save up to 30% water 
            and optimizes plant care in agriculture and gardening.
            """
        },
        {
            "title": "Autonomous Cleaning Robot",
            "text": """
            A cleaning robot that uses LIDAR technology and AI to map spaces,
            avoid obstacles, and clean efficiently. The robot can vacuum, mop floors
            and automatically return to charging station. Applications in homes and offices,
            reducing manual cleaning time by 80%.
            """
        }
    ]
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"🔬 DEMO {i}: {sample['title']}")
        print(f"{'='*80}")
        
        results = extractor.extract_keywords(sample['text'])
        
        print(f"\n📋 Results for '{sample['title']}':")
        print(json.dumps(results, indent=2, ensure_ascii=False))


def interactive_mode():
    """Interactive mode for user input"""
    
    # Use simple mode without checkpointer to avoid thread_id requirements
    # extractor = CoreConceptExtractor(model_name="llama3", use_checkpointer=False)
    extractor = CoreConceptExtractor(model_name="qwen2.5:0.5b-instruct", use_checkpointer=False)

    print("🚀 WELCOME TO PATENT SEED KEYWORD EXTRACTION SYSTEM")
    print("="*70)
    print("✨ New Workflow: Automatic refinement with final human evaluation")
    print("   • Phases 1-3 run automatically")
    print("   • You evaluate final results and can approve, edit, or re-run")
    
    while True:
        print("\nPlease enter your idea description or technical document:")
        print("(Type 'quit' to exit)")
        
        user_input = input("\n📝 Content: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("👋 Thank you for using the system!")
            break
        
        if not user_input:
            print("⚠️ Please enter valid content!")
            continue
        
        try:
            print("\n🔄 Processing through all 3 phases...")
            results = extractor.extract_keywords(user_input)
            
            print("\n" + "="*60)
            print("📊 EXTRACTION RESULTS")
            print("="*60)
            
            if results['final_keywords']:
                print("\n🔑 Final seed keywords:")
                for category, keywords in results['final_keywords'].items():
                    category_name = category.replace('_', ' ').title()
                    print(f"  • {category_name}: {keywords}")
            
            print(f"\n📝 Processing history:")
            for msg in results['messages']:
                print(f"  → {msg}")
            
            # Handle user action if they chose to re-run
            user_action = results.get('user_action')
            if user_action == 'rerun':
                print("\n🔄 Re-running with your feedback...")
                # The re-run is handled within the workflow
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again with different content.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_with_sample_patents()
    else:
        interactive_mode()
