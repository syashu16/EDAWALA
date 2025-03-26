"""
Generates insights and stories from data
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import os
import logging
from datetime import datetime

# Import internal modules
from .stat_insights import get_statistical_insights
from .text_generator import format_insights_as_story, format_insights_as_summary

# Add proper logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_insights(
    df: pd.DataFrame,
    use_llm: bool = False,
    llm_provider: Optional[str] = 'gemini',
    max_insights: int = 15
) -> List[Dict[str, Any]]:
    """
    Generate insights from a DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to analyze
    use_llm : bool, optional
        Whether to use LLM-enhanced insights (default: False)
    llm_provider : str, optional
        LLM provider to use ('gemini' or 'openai') (default: 'gemini')
    max_insights : int, optional
        Maximum number of insights to generate (default: 15)
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of insights, each as a dictionary
    """
    # First get statistical insights
    insights = get_statistical_insights(df, max_insights=max_insights)
    
    # Enhance with LLM if requested and possible
    if use_llm:
        try:
            # Check for API keys
            if llm_provider == 'gemini' and not os.environ.get('GOOGLE_API_KEY'):
                logger.warning("No Google API key found, using statistical insights only")
                return insights
            elif llm_provider == 'openai' and not os.environ.get('OPENAI_API_KEY'):
                logger.warning("No OpenAI API key found, using statistical insights only")
                return insights
            
            # Import appropriate module based on provider
            if llm_provider == 'gemini':
                from .llm_insights_gemini import enhance_insights_with_llm
            else:
                from .llm_insights_openai import enhance_insights_with_llm
                
            # Enhance insights
            enhanced_insights = enhance_insights_with_llm(df, insights)
            return enhanced_insights
            
        except Exception as e:
            logger.error(f"Error enhancing insights with LLM: {e}")
            logger.info("Falling back to statistical insights")
            return insights
    
    return insights

def generate_story(
    df: pd.DataFrame,
    insights: Optional[List[Dict[str, Any]]] = None,
    use_llm: bool = False,
    llm_provider: Optional[str] = 'gemini'
) -> str:
    """
    Generate a data story from insights.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to analyze
    insights : List[Dict[str, Any]], optional
        List of insights to use (if None, will generate them)
    use_llm : bool, optional
        Whether to use LLM-enhanced story (default: False)
    llm_provider : str, optional
        LLM provider to use ('gemini' or 'openai') (default: 'gemini')
        
    Returns:
    --------
    str
        Formatted data story
    """
    # Generate insights if not provided
    if insights is None:
        insights = generate_insights(df, use_llm=use_llm, llm_provider=llm_provider)
    
    # Format insights as a story
    story = format_insights_as_story(df, insights, use_llm, llm_provider)
    
    return story

def get_executive_summary(
    df: pd.DataFrame,
    insights: Optional[List[Dict[str, Any]]] = None,
    use_llm: bool = False,
    llm_provider: Optional[str] = 'gemini'
) -> str:
    """
    Generate an executive summary from insights.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to analyze
    insights : List[Dict[str, Any]], optional
        List of insights to use (if None, will generate them)
    use_llm : bool, optional
        Whether to use LLM-enhanced summary (default: False)
    llm_provider : str, optional
        LLM provider to use ('gemini' or 'openai') (default: 'gemini')
        
    Returns:
    --------
    str
        Formatted executive summary
    """
    # Generate insights if not provided
    if insights is None:
        insights = generate_insights(df, use_llm=use_llm, llm_provider=llm_provider)
    
    # Format insights as a summary
    summary = format_insights_as_summary(df, insights, use_llm, llm_provider)
    
    return summary