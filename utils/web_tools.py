import requests
import yfinance as yf
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, Optional
import time

class DataCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def collect_company_info(self, company_name: str) -> Dict:
        """Collect comprehensive company information"""
        data = {
            "company_name": company_name,
            "financial_data": self._get_financial_data(company_name),
            "company_profile": self._get_company_profile(company_name),
            "recent_news": self._get_recent_news(company_name),
            "stock_performance": self._get_stock_performance(company_name)
        }
        return data
    
    def _get_financial_data(self, company_name: str) -> Dict:
        """Get financial data using yfinance"""
        try:
            # Try to find ticker symbol
            ticker = self._find_ticker_symbol(company_name)
            if not ticker:
                return {"error": "Ticker symbol not found"}
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "ticker": ticker,
                "market_cap": info.get("marketCap"),
                "revenue": info.get("totalRevenue"),
                "employees": info.get("fullTimeEmployees"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "pe_ratio": info.get("trailingPE"),
                "profit_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth")
            }
        except Exception as e:
            return {"error": f"Failed to get financial data: {str(e)}"}
    
    def _find_ticker_symbol(self, company_name: str) -> Optional[str]:
        """Attempt to find ticker symbol for company"""
        # Simple mapping for common companies
        ticker_map = {
            "apple": "AAPL",
            "microsoft": "MSFT", 
            "google": "GOOGL",
            "amazon": "AMZN",
            "tesla": "TSLA",
            "meta": "META",
            "netflix": "NFLX"
        }
        
        company_lower = company_name.lower()
        for key, ticker in ticker_map.items():
            if key in company_lower:
                return ticker
        
        return None
    
    def _get_company_profile(self, company_name: str) -> Dict:
        """Get company profile information"""
        # This would typically scrape company websites or use APIs
        # For demo purposes, return basic structure
        return {
            "description": f"Profile information for {company_name}",
            "headquarters": "Unknown",
            "founded": "Unknown",
            "ceo": "Unknown"
        }
    
    def _get_recent_news(self, company_name: str) -> list:
        """Get recent news about the company"""
        # This would typically use news APIs
        return [
            {"title": f"Recent news about {company_name}", "date": "2024-01-01"}
        ]
    
    def _get_stock_performance(self, company_name: str) -> Dict:
        """Get stock performance metrics"""
        ticker = self._find_ticker_symbol(company_name)
        if not ticker:
            return {"error": "No stock data available"}
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            return {
                "52_week_high": hist['High'].max(),
                "52_week_low": hist['Low'].min(),
                "current_price": hist['Close'].iloc[-1],
                "ytd_return": ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
            }
        except Exception as e:
            return {"error": f"Failed to get stock performance: {str(e)}"}