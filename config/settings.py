import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SERP_API_KEY = os.getenv("SERP_API_KEY")
    
    # Model configuration
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")  # Default fallback
    
    # Analysis parameters
    MAX_COMPETITORS = 5
    ANALYSIS_DEPTH = "comprehensive"
    
    # Data sources
    FINANCIAL_SOURCES = [
        "yahoo_finance",
        "sec_filings", 
        "company_websites",
        "industry_reports"
    ]
    
    # Output settings
    SAVE_RESULTS = True
    OUTPUT_FORMAT = "json"