from agents.market_analyzer import MarketAnalyzer
from config.settings import Config
import json
from datetime import datetime

def main():
    print("🔍 Market Analysis Agent")
    print("=" * 50)
    
    # Check API key
    if not Config.OPENAI_API_KEY:
        print("❌ Error: OpenAI API key not found. Please check your .env file.")
        return
    
    # Check which models are available and select one
    available_models = [
        "gpt-4o-mini",
        "gpt-3.5-turbo", 
        "gpt-4-turbo-preview",
        "gpt-4"
    ]
    
    selected_model = None
    for model in available_models:
        try:
            # Test the model with a simple request
            test_llm = ChatOpenAI(model=model, api_key=Config.OPENAI_API_KEY)
            test_llm.invoke([HumanMessage("Hello")])
            selected_model = model
            print(f"✅ Using model: {model}")
            break
        except Exception as e:
            print(f"❌ Model {model} not available: {str(e)}")
            continue
    
    if not selected_model:
        print("❌ No available models found. Please check your OpenAI subscription.")
        return
    
    # Initialize analyzer with working model
    analyzer = MarketAnalyzer(Config.OPENAI_API_KEY, selected_model)
    
    # Get company to analyze
    company_name = input("\n📊 Enter company name to analyze: ").strip()
    
    if not company_name:
        print("❌ Company name is required.")
        return
    
    print(f"\n🚀 Starting market analysis for: {company_name}")
    print("This may take a few minutes...\n")
    
    try:
        # Run analysis
        result = analyzer.analyze_company(company_name)
        
        # Display results (rest of the code remains the same)
        print("✅ Analysis Complete!")
        print("=" * 50)
        
        print(f"\n📈 MARKET ANALYSIS REPORT FOR {company_name.upper()}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model Used: {selected_model}")
        print(f"Confidence Score: {result.get('confidence_score', 'N/A')}/10")
        
        print(f"\n🏭 INDUSTRY: {result.get('industry', 'Not identified')}")
        
        print(f"\n🎯 COMPETITORS IDENTIFIED: {len(result.get('competitors', []))}")
        for i, comp in enumerate(result.get('competitors', [])[:5], 1):
            print(f"  {i}. {comp.get('name', 'Unknown')} ({comp.get('threat_level', 'Unknown')} threat)")
        
        print(f"\n📊 MARKET POSITION:")
        position = result.get('market_position', {})
        print(f"  Position: {position.get('position', 'Unknown')}")
        print(f"  Market Share: {position.get('market_share', 'Unknown')}")
        
        print(f"\n📈 KEY TRENDS: {len(result.get('market_trends', []))}")
        for trend in result.get('market_trends', [])[:3]:
            print(f"  • {trend.get('trend', 'Unknown trend')} ({trend.get('impact', 'Unknown')} impact)")
        
        if result.get('final_report'):
            print(f"\n📄 EXECUTIVE SUMMARY:")
            print("-" * 30)
            report_preview = result['final_report'][:500] + "..." if len(result['final_report']) > 500 else result['final_report']
            print(report_preview)
        
        # Save results
        if Config.SAVE_RESULTS:
            filename = f"market_analysis_{company_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 Full results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print(f"💡 Tip: Make sure your OpenAI API key is valid and has sufficient credits.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()