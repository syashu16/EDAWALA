"""
Google Gemini-powered insight enhancement for EDAwala
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os
import logging
import time
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def enhance_insights_with_llm(df: pd.DataFrame, insights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhance insights using Google's Gemini LLM.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame being analyzed
    insights : List[Dict[str, Any]]
        Statistical insights to enhance
        
    Returns:
    --------
    List[Dict[str, Any]]
        Enhanced insights
    """
    try:
        # Check if API key is available
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            logger.warning("No Google API key found in environment variables")
            return insights
        
        # Import Gemini
        import google.generativeai as genai
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Get a model
        model = genai.GenerativeModel('gemini-pro')
        
        # Prepare dataset description
        df_head = df.head(5).to_string()
        df_info = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
        df_cols = ", ".join(df.columns)
        
        # Create enhanced insights
        enhanced_insights = []
        
        for insight in insights:
            # Create prompt for enhancing the insight
            prompt = f"""
            You are an expert data analyst enhancing the description of a data insight.
            
            Dataset Information:
            {df_info}
            Columns: {df_cols}
            
            Sample data:
            {df_head}
            
            Original Insight:
            Title: {insight['title']}
            Description: {insight['description']}
            
            Please enhance this insight by:
            1. Making the description more detailed and insightful
            2. Adding business implications where appropriate
            3. Suggesting a specific recommendation based on this insight
            
            Respond ONLY with a JSON object with the following structure:
            {{
                "title": "Enhanced title",
                "description": "Enhanced description",
                "recommendation": "A specific recommendation"
            }}
            
            No other text before or after the JSON.
            """
            
            try:
                # Generate enhanced insight
                response = model.generate_content(prompt)
                
                # Extract JSON from response
                try:
                    json_text = response.text.strip()
                    if json_text.startswith('```json'):
                        json_text = json_text.replace('```json', '').replace('```', '')
                    
                    enhanced = json.loads(json_text)
                    
                    # Update the insight
                    enhanced_insight = insight.copy()
                    enhanced_insight['title'] = enhanced.get('title', insight['title'])
                    enhanced_insight['description'] = enhanced.get('description', insight['description'])
                    
                    # Add recommendation if provided
                    if 'recommendation' in enhanced:
                        enhanced_insight['recommendation'] = enhanced['recommendation']
                        
                    enhanced_insights.append(enhanced_insight)
                except json.JSONDecodeError as je:
                    logger.warning(f"Failed to parse JSON from response: {response.text}")
                    logger.warning(f"JSON error: {je}")
                    enhanced_insights.append(insight)
                    
            except Exception as e:
                logger.warning(f"Error enhancing insight: {e}")
                enhanced_insights.append(insight)
                
            # Sleep to avoid rate limits
            time.sleep(1)
        
        return enhanced_insights
        
    except Exception as e:
        logger.error(f"Error in enhancing insights with Gemini: {e}")
        # Return original insights if enhancement fails
        return insights

def generate_story_with_llm(df: pd.DataFrame, insights: List[Dict[str, Any]]) -> str:
    """
    Generate a data story using Google's Gemini LLM.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame being analyzed
    insights : List[Dict[str, Any]]
        List of insights
        
    Returns:
    --------
    str
        Formatted data story
    """
    try:
        # Check if API key is available
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            logger.warning("No Google API key found in environment variables")
            from .text_generator import generate_basic_story
            return generate_basic_story(df, insights)
        
        # Import Gemini
        import google.generativeai as genai
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Get a model
        model = genai.GenerativeModel('gemini-pro')
        
        # Prepare dataset description
        df_head = df.head(5).to_string()
        df_info = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
        df_cols = ", ".join(df.columns)
        
        # Format insights for prompt
        insights_text = ""
        for i, insight in enumerate(insights):
            insights_text += f"Insight {i+1}: {insight['title']}\n"
            insights_text += f"Description: {insight['description']}\n"
            if 'recommendation' in insight:
                insights_text += f"Recommendation: {insight['recommendation']}\n"
            insights_text += "\n"
        
        # Create prompt
        prompt = f"""
        You are an expert data storyteller creating a comprehensive narrative based on data insights.
        
        Dataset Information:
        {df_info}
        Columns: {df_cols}
        
        Sample data:
        {df_head}
        
        Insights:
        {insights_text}
        
        Please create a well-structured data story that:
        1. Has a clear narrative flow with introduction, body, and conclusion
        2. Groups related insights by themes
        3. Explains the business or practical implications
        4. Includes section headers and proper formatting
        5. Provides actionable recommendations based on the insights
        
        Format the story with Markdown, using headers, bullet points, and emphasis where appropriate.
        """
        
        # Generate story
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating story with Gemini: {e}")
        # Fall back to basic story generation
        from .text_generator import generate_basic_story
        return generate_basic_story(df, insights)

def generate_summary_with_llm(df: pd.DataFrame, insights: List[Dict[str, Any]]) -> str:
    """
    Generate an executive summary using Google's Gemini LLM.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame being analyzed
    insights : List[Dict[str, Any]]
        List of insights
        
    Returns:
    --------
    str
        Executive summary
    """
    try:
        # Check if API key is available
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            logger.warning("No Google API key found in environment variables")
            from .text_generator import generate_basic_summary
            return generate_basic_summary(df, insights)
        
        # Import Gemini
        import google.generativeai as genai
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Get a model
        model = genai.GenerativeModel('gemini-pro')
        
        # Prepare dataset description
        df_head = df.head(5).to_string()
        df_info = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
        df_cols = ", ".join(df.columns)
        
        # Format insights for prompt
        insights_text = ""
        for i, insight in enumerate(insights[:10]):  # Limit to top 10 insights
            insights_text += f"Insight {i+1}: {insight['title']}\n"
            insights_text += f"Description: {insight['description']}\n"
            if 'recommendation' in insight:
                insights_text += f"Recommendation: {insight['recommendation']}\n"
            insights_text += "\n"
        
        # Create prompt
        prompt = f"""
        You are an expert data analyst creating a concise executive summary based on data insights.
        
        Dataset Information:
        {df_info}
        Columns: {df_cols}
        
        Sample data:
        {df_head}
        
        Insights:
        {insights_text}
        
        Please create a concise executive summary (2-3 paragraphs) that:
        1. Highlights the most important findings
        2. Focuses on business or practical implications
        3. Provides 2-3 key recommendations
        
        The summary should be direct, data-driven, and actionable for decision-makers.
        """
        
        # Generate summary
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating summary with Gemini: {e}")
        # Fall back to basic summary generation
        from .text_generator import generate_basic_summary
        return generate_basic_summary(df, insights)