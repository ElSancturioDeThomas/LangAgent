from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
import pandas as pd
from datetime import datetime
from config.settings import Config

class MarketAnalysisState(TypedDict):
    target_company: str
    industry: Optional[str]
    competitors: List[Dict]
    company_data: Dict
    market_position: Dict
    swot_analysis: Dict
    financial_comparison: Dict
    market_trends: List[Dict]
    final_report: Optional[str]
    confidence_score: Optional[float]

from config.settings import Config

class MarketAnalyzer:
    def __init__(self, api_key: str, model: str = None):
        self.llm = ChatOpenAI(
            model=model or Config.OPENAI_MODEL,
            api_key=api_key,
            temperature=0.1
        )
        self.workflow = self._build_workflow()
        print(f"✅ Initialized with model: {model or Config.OPENAI_MODEL}")
    
    def _build_workflow(self) -> StateGraph:
        """Build the market analysis workflow"""
        workflow = StateGraph(MarketAnalysisState)
        
        # Add nodes
        workflow.add_node("identify_industry", self._identify_industry)
        workflow.add_node("find_competitors", self._find_competitors)
        workflow.add_node("collect_company_data", self._collect_company_data)
        workflow.add_node("analyze_financials", self._analyze_financials)
        workflow.add_node("perform_swot", self._perform_swot_analysis)
        workflow.add_node("assess_market_position", self._assess_market_position)
        workflow.add_node("identify_trends", self._identify_market_trends)
        workflow.add_node("generate_report", self._generate_final_report)
        
        # Define workflow edges
        workflow.set_entry_point("identify_industry")
        workflow.add_edge("identify_industry", "find_competitors")
        workflow.add_edge("find_competitors", "collect_company_data")
        workflow.add_edge("collect_company_data", "analyze_financials")
        workflow.add_edge("analyze_financials", "perform_swot")
        workflow.add_edge("perform_swot", "assess_market_position")
        workflow.add_edge("assess_market_position", "identify_trends")
        workflow.add_edge("identify_trends", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    def _identify_industry(self, state: MarketAnalysisState) -> MarketAnalysisState:
        """Identify the industry and market segment"""
        prompt = f"""
        Analyze the company "{state['target_company']}" and identify:
        1. Primary industry classification (NAICS/SIC codes if possible)
        2. Market segment and sub-segments
        3. Business model type
        4. Key value propositions
        5. Target customer segments
        
        Provide a structured analysis focusing on market categorization.
        """
        
        response = self.llm.invoke([
            SystemMessage("You are an industry classification expert with deep knowledge of market segments."),
            HumanMessage(prompt)
        ])
        
        # Extract industry information
        industry_info = self._extract_industry_info(response.content)
        
        return {
            **state,
            "industry": industry_info.get("primary_industry"),
            "company_data": {
                "industry_classification": industry_info,
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
    
    def _find_competitors(self, state: MarketAnalysisState) -> MarketAnalysisState:
        """Identify and categorize competitors"""
        prompt = f"""
        For {state['target_company']} in the {state['industry']} industry, identify:
        
        DIRECT COMPETITORS (5-7 companies):
        - Companies with similar products/services
        - Similar target markets
        - Similar business models
        
        INDIRECT COMPETITORS (3-5 companies):
        - Companies solving similar customer problems differently
        - Adjacent market players
        
        EMERGING THREATS (2-3 companies):
        - Startups or new entrants
        - Companies from adjacent industries expanding
        
        For each competitor, provide:
        - Company name
        - Market cap/size (if public)
        - Key differentiators
        - Competitive threat level (High/Medium/Low)
        """
        
        response = self.llm.invoke([
            SystemMessage("You are a competitive intelligence analyst with expertise in market mapping."),
            HumanMessage(prompt)
        ])
        
        competitors = self._extract_competitors(response.content)
        
        return {
            **state,
            "competitors": competitors
        }
    
    def _collect_company_data(self, state: MarketAnalysisState) -> MarketAnalysisState:
        """Collect comprehensive data on target company and competitors"""
        from utils.web_tools import DataCollector
        
        collector = DataCollector()
        
        # Collect data for target company
        target_data = collector.collect_company_info(state['target_company'])
        
        # Collect data for top competitors
        competitor_data = []
        for competitor in state['competitors'][:5]:  # Limit to top 5
            comp_data = collector.collect_company_info(competitor['name'])
            competitor_data.append({
                **competitor,
                "financial_data": comp_data
            })
        
        return {
            **state,
            "company_data": {
                **state['company_data'],
                "target_company_data": target_data,
                "competitor_data": competitor_data
            }
        }
    
    def _analyze_financials(self, state: MarketAnalysisState) -> MarketAnalysisState:
        """Perform financial comparison and analysis"""
        prompt = f"""
        Analyze the financial performance comparison:
        
        Target Company: {state['target_company']}
        Data: {state['company_data'].get('target_company_data', {})}
        
        Competitors: {[comp['name'] for comp in state['competitors'][:3]]}
        
        Provide analysis on:
        1. Revenue comparison and growth rates
        2. Profitability metrics
        3. Market valuation multiples
        4. Financial health indicators
        5. Investment in R&D and growth
        
        Identify financial strengths and weaknesses relative to competitors.
        """
        
        response = self.llm.invoke([
            SystemMessage("You are a financial analyst specializing in competitive benchmarking."),
            HumanMessage(prompt)
        ])
        
        financial_analysis = self._extract_financial_analysis(response.content)
        
        return {
            **state,
            "financial_comparison": financial_analysis
        }
    
    def _perform_swot_analysis(self, state: MarketAnalysisState) -> MarketAnalysisState:
        """Conduct comprehensive SWOT analysis"""
        prompt = f"""
        Conduct a detailed SWOT analysis for {state['target_company']}:
        
        Context:
        - Industry: {state['industry']}
        - Competitors: {[comp['name'] for comp in state['competitors'][:5]]}
        - Financial position: {state['financial_comparison']}
        
        STRENGTHS:
        - Internal capabilities and advantages
        - Competitive advantages
        - Resources and assets
        
        WEAKNESSES:
        - Internal limitations
        - Areas needing improvement
        - Competitive disadvantages
        
        OPPORTUNITIES:
        - Market trends and growth areas
        - Untapped markets
        - Strategic partnerships potential
        
        THREATS:
        - Competitive pressures
        - Market risks
        - Regulatory/economic challenges
        
        Provide specific, actionable insights for each category.
        """
        
        response = self.llm.invoke([
            SystemMessage("You are a strategic business analyst with expertise in SWOT methodology."),
            HumanMessage(prompt)
        ])
        
        swot = self._extract_swot_analysis(response.content)
        
        return {
            **state,
            "swot_analysis": swot
        }
    
    def _assess_market_position(self, state: MarketAnalysisState) -> MarketAnalysisState:
        """Assess overall market position and competitive standing"""
        prompt = f"""
        Assess {state['target_company']}'s market position:
        
        Based on:
        - Competitor landscape: {len(state['competitors'])} identified competitors
        - Financial performance: {state['financial_comparison']}
        - SWOT analysis: {state['swot_analysis']}
        
        Determine:
        1. Market share position (Leader/Challenger/Follower/Niche)
        2. Competitive positioning quadrant
        3. Strategic group membership
        4. Barriers to entry in their market
        5. Threat of substitutes
        6. Bargaining power (suppliers/customers)
        
        Provide Porter's Five Forces analysis and strategic recommendations.
        """
        
        response = self.llm.invoke([
            SystemMessage("You are a strategy consultant specializing in competitive positioning."),
            HumanMessage(prompt)
        ])
        
        market_position = self._extract_market_position(response.content)
        
        return {
            **state,
            "market_position": market_position
        }
    
    def _identify_market_trends(self, state: MarketAnalysisState) -> MarketAnalysisState:
        """Identify key market trends and future outlook"""
        prompt = f"""
        Identify key market trends affecting {state['target_company']} in {state['industry']}:
        
        Analyze:
        1. Technology trends impacting the industry
        2. Consumer behavior changes
        3. Regulatory developments
        4. Economic factors
        5. Competitive dynamics evolution
        6. Market growth projections
        
        For each trend, assess:
        - Impact level (High/Medium/Low)
        - Time horizon (Short/Medium/Long term)
        - Strategic implications for {state['target_company']}
        """
        
        response = self.llm.invoke([
            SystemMessage("You are a market research analyst specializing in trend analysis and forecasting."),
            HumanMessage(prompt)
        ])
        
        trends = self._extract_market_trends(response.content)
        
        return {
            **state,
            "market_trends": trends
        }
    
    def _generate_final_report(self, state: MarketAnalysisState) -> MarketAnalysisState:
        """Generate comprehensive final report"""
        prompt = f"""
        Create an executive summary report for {state['target_company']} market analysis:
        
        EXECUTIVE SUMMARY:
        - Company overview and market position
        - Key findings and insights
        - Strategic recommendations
        
        COMPETITIVE LANDSCAPE:
        - Industry overview: {state['industry']}
        - Competitor analysis: {len(state['competitors'])} competitors identified
        - Market positioning: {state['market_position']}
        
        FINANCIAL PERFORMANCE:
        - Competitive benchmarking results
        - Key financial metrics comparison
        
        STRATEGIC ANALYSIS:
        - SWOT summary with priority items
        - Market trends impact assessment
        - Growth opportunities identification
        
        RECOMMENDATIONS:
        - Top 3 strategic priorities
        - Risk mitigation strategies
        - Market expansion opportunities
        
        Confidence Score: Assess analysis confidence (1-10 scale)
        """
        
        response = self.llm.invoke([
            SystemMessage("You are a senior management consultant creating executive-level deliverables."),
            HumanMessage(prompt)
        ])
        
        # Extract confidence score
        confidence = self._extract_confidence_score(response.content)
        
        return {
            **state,
            "final_report": response.content,
            "confidence_score": confidence
        }
    
    def analyze_company(self, company_name: str) -> Dict:
        """Main method to analyze a company"""
        initial_state = {
            "target_company": company_name,
            "industry": None,
            "competitors": [],
            "company_data": {},
            "market_position": {},
            "swot_analysis": {},
            "financial_comparison": {},
            "market_trends": [],
            "final_report": None,
            "confidence_score": None
        }
        
        result = self.workflow.invoke(initial_state)
        return result
    
    # Helper methods for data extraction
    def _extract_industry_info(self, content: str) -> Dict:
        """Extract structured industry information from LLM response"""
        # Implementation would parse the LLM response
        # For now, return a basic structure
        return {
            "primary_industry": "Technology",
            "sub_segments": [],
            "business_model": "SaaS",
            "naics_code": None
        }
    
    def _extract_competitors(self, content: str) -> List[Dict]:
        """Extract competitor list from LLM response"""
        # Implementation would parse competitor information
        return [
            {"name": "Competitor 1", "threat_level": "High", "type": "direct"},
            {"name": "Competitor 2", "threat_level": "Medium", "type": "indirect"}
        ]
    
    def _extract_financial_analysis(self, content: str) -> Dict:
        """Extract financial analysis from LLM response"""
        return {"revenue_growth": "15%", "profitability": "Above average"}
    
    def _extract_swot_analysis(self, content: str) -> Dict:
        """Extract SWOT analysis from LLM response"""
        return {
            "strengths": [],
            "weaknesses": [],
            "opportunities": [],
            "threats": []
        }
    
    def _extract_market_position(self, content: str) -> Dict:
        """Extract market positioning from LLM response"""
        return {"position": "Challenger", "market_share": "15%"}
    
    def _extract_market_trends(self, content: str) -> List[Dict]:
        """Extract market trends from LLM response"""
        return [{"trend": "AI adoption", "impact": "High", "timeline": "Short-term"}]
    
    def _extract_confidence_score(self, content: str) -> float:
        """Extract confidence score from LLM response"""
        return 7.5