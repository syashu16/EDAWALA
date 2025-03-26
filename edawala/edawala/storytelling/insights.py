"""
Main module for EDAwala's Storytelling EDA feature.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
import logging
import os

from .stat_insights import StatisticalInsights
from .text_generator import TextGenerator
from .llm_insights import LLMInsightsGenerator

logger = logging.getLogger(__name__)

class InsightGenerator:
    """
    Main class for generating insights and stories from data.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        use_llm: bool = False,
        api_key: Optional[str] = None,
        llm_provider: str = "gemini"  # Default to Gemini
    ):
        """
        Initialize the insight generator.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to analyze
        use_llm : bool, optional
            Whether to use LLM for enhanced insights (default: False)
        api_key : str, optional
            API key for the LLM provider (default: None, will use environment variable)
        llm_provider : str, optional
            LLM provider to use: "gemini" or "openai" (default: "gemini")
        """
        self.df = df
        self.use_llm = use_llm
        self.api_key = api_key
        self.llm_provider = llm_provider.lower()
        
        # Set API key from environment if not provided
        if not self.api_key:
            if self.llm_provider == "gemini":
                self.api_key = os.environ.get("GOOGLE_API_KEY")
            else:
                self.api_key = os.environ.get("OPENAI_API_KEY")
        
        # Initialize components
        self.stat_insights = StatisticalInsights(df)
        self.text_generator = TextGenerator()
        
        if use_llm:
            self.llm_generator = LLMInsightsGenerator(api_key=self.api_key, provider=self.llm_provider)
        else:
            self.llm_generator = None
    
    def generate_insights(self, max_insights: int = 20) -> List[Dict[str, Any]]:
        """
        Generate insights from the data.
        
        Parameters:
        -----------
        max_insights : int, optional
            Maximum number of insights to return (default: 20)
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of insights
        """
        # Get statistical insights
        insights = self.stat_insights.get_all_insights()
        
        # Get LLM insights if enabled
        if self.use_llm and self.llm_generator:
            try:
                llm_insights = self.llm_generator.generate_llm_insights(self.df, insights)
                insights.extend(llm_insights)
            except Exception as e:
                logger.error(f"Error generating LLM insights: {str(e)}")
        
        # Sort by importance
        insights.sort(key=lambda x: x.get('importance', 0), reverse=True)
        
        # Return top insights
        return insights[:max_insights]
    
    def generate_story(self, insights: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a data story from insights.
        
        Parameters:
        -----------
        insights : List[Dict[str, Any]], optional
            List of insights (if None, will generate insights)
            
        Returns:
        --------
        str
            A markdown-formatted data story
        """
        if insights is None:
            insights = self.generate_insights()
            
        # If LLM is enabled, use it for story generation
        if self.use_llm and self.llm_generator:
            try:
                llm_story = self.llm_generator.generate_data_story(self.df, insights)
                if llm_story:
                    return llm_story
            except Exception as e:
                logger.error(f"Error generating LLM story: {str(e)}")
                
        # Fallback to rule-based story generation
        return self.text_generator.generate_story(self.df, insights)
    
    def get_executive_summary(self, max_length: int = 500) -> str:
        """
        Generate a concise executive summary of key findings.
        
        Parameters:
        -----------
        max_length : int, optional
            Maximum length in characters (default: 500)
            
        Returns:
        --------
        str
            Executive summary
        """
        # Get top insights
        insights = self.generate_insights(max_insights=5)
        
        # If LLM is enabled, use it for executive summary
        if self.use_llm and self.llm_generator:
            try:
                prompt = f"""
                Create a concise executive summary (max 3-4 sentences) of these key data findings:
                
                - {insights[0].get('description', '')}
                - {insights[1].get('description', '')}
                - {insights[2].get('description', '')}
                
                Focus only on the most important insights that executives need to know.
                """
                
                summary = self.llm_generator.generate_text(prompt, temperature=0.5, max_tokens=150)
                
                if summary and len(summary) <= max_length:
                    return summary
                    
            except Exception as e:
                logger.error(f"Error generating executive summary: {str(e)}")
                
        # Fallback to rule-based summary
        # Sort insights by importance
        insights.sort(key=lambda x: x.get('importance', 0), reverse=True)
        
        # Get descriptions of top insights
        descriptions = [insight.get('description', '') for insight in insights[:3]]
        
        # Create summary
        summary = "Key findings: " + " ".join(descriptions)
        
        # Truncate if needed
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
            
        return summary


def generate_insights(
    df: pd.DataFrame,
    use_llm: bool = False,
    api_key: Optional[str] = None,
    llm_provider: str = "gemini",
    max_insights: int = 20
) -> List[Dict[str, Any]]:
    """
    Generate insights from a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    use_llm : bool, optional
        Whether to use LLM for enhanced insights (default: False)
    api_key : str, optional
        API key for the LLM provider (default: None, will use environment variable)
    llm_provider : str, optional
        LLM provider to use: "gemini" or "openai" (default: "gemini")
    max_insights : int, optional
        Maximum number of insights to return (default: 20)
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of insights
    """
    generator = InsightGenerator(df, use_llm, api_key, llm_provider)
    return generator.generate_insights(max_insights)


def generate_story(
    df: pd.DataFrame,
    insights: Optional[List[Dict[str, Any]]] = None,
    use_llm: bool = False,
    api_key: Optional[str] = None,
    llm_provider: str = "gemini"
) -> str:
    """
    Generate a data story from a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    insights : List[Dict[str, Any]], optional
        List of insights (if None, will generate insights)
    use_llm : bool, optional
        Whether to use LLM for enhanced insights (default: False)
    api_key : str, optional
        API key for the LLM provider (default: None, will use environment variable)
    llm_provider : str, optional
        LLM provider to use: "gemini" or "openai" (default: "gemini")
        
    Returns:
    --------
    str
        A markdown-formatted data story
    """
    generator = InsightGenerator(df, use_llm, api_key, llm_provider)
    return generator.generate_story(insights)


def get_executive_summary(
    df: pd.DataFrame,
    use_llm: bool = False,
    api_key: Optional[str] = None,
    llm_provider: str = "gemini",
    max_length: int = 500
) -> str:
    """
    Generate an executive summary from a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    use_llm : bool, optional
        Whether to use LLM for enhanced insights (default: False)
    api_key : str, optional
        API key for the LLM provider (default: None, will use environment variable)
    llm_provider : str, optional
        LLM provider to use: "gemini" or "openai" (default: "gemini")
    max_length : int, optional
        Maximum length in characters (default: 500)
        
    Returns:
    --------
    str
        Executive summary
    """
    generator = InsightGenerator(df, use_llm, api_key, llm_provider)
    return generator.get_executive_summary(max_length)