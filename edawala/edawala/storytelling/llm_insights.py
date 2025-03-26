"""
LLM-powered insights generator for EDAwala's Storytelling module.
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Union, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class LLMInsightsGenerator:
    """
    Generates advanced insights using language model APIs.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: str = "gemini-pro", 
        provider: str = "gemini"
    ):
        """
        Initialize the LLM insights generator.
        
        Parameters:
        -----------
        api_key : str, optional
            API key for the chosen provider (defaults to environment variable)
        model : str, optional
            Model to use (default: "gemini-pro" for Gemini, "gpt-3.5-turbo" for OpenAI)
        provider : str, optional
            LLM provider to use: "gemini" or "openai" (default: "gemini")
        """
        self.provider = provider.lower()
        self.model = model
        
        # Set default models based on provider if not specified
        if self.provider == "gemini" and model == "gpt-3.5-turbo":
            self.model = "gemini-pro"
        elif self.provider == "openai" and model == "gemini-pro":
            self.model = "gpt-3.5-turbo"
            
        # Handle API keys
        if api_key:
            self.api_key = api_key
        else:
            # Try to get from environment variables
            if self.provider == "gemini":
                self.api_key = os.environ.get("GOOGLE_API_KEY")
            else:
                self.api_key = os.environ.get("OPENAI_API_KEY")
        
        # Check if provider is available
        self.is_available = self._check_provider()
        
    def _check_provider(self) -> bool:
        """
        Check if the selected LLM provider is available.
        
        Returns:
        --------
        bool
            True if provider is available, False otherwise
        """
        try:
            if self.provider == "gemini":
                try:
                    import google.generativeai
                    return True
                except ImportError:
                    logger.warning("Google GenerativeAI package not found. Install with `pip install google-generativeai` for Gemini integration.")
                    return False
            else:  # openai
                try:
                    import openai
                    return True
                except ImportError:
                    logger.warning("OpenAI package not found. Install with `pip install openai` for OpenAI integration.")
                    return False
        except Exception as e:
            logger.error(f"Error checking provider availability: {str(e)}")
            return False
    
    def _generate_with_gemini(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generate text using Google's Gemini API.
        
        Parameters:
        -----------
        prompt : str
            The prompt to send to the model
        temperature : float, optional
            Creativity parameter (default: 0.7)
        max_tokens : int, optional
            Maximum response length (default: 1000)
            
        Returns:
        --------
        str
            Generated text
        """
        try:
            import google.generativeai as genai
            
            # Configure API
            genai.configure(api_key=self.api_key)
            
            # Set up generation config
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.9,
            }
            
            # Generate content
            model = genai.GenerativeModel(model_name=self.model, generation_config=generation_config)
            response = model.generate_content(prompt)
            
            # Return text
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating with Gemini: {str(e)}")
            return ""
    
    def _generate_with_openai(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generate text using OpenAI's API.
        
        Parameters:
        -----------
        prompt : str
            The prompt to send to the model
        temperature : float, optional
            Creativity parameter (default: 0.7)
        max_tokens : int, optional
            Maximum response length (default: 1000)
            
        Returns:
        --------
        str
            Generated text
        """
        try:
            import openai
            
            # Set API key
            if self.api_key:
                openai.api_key = self.api_key
                
            # Make the API call
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data analysis assistant that provides clear, specific insights from data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Return text
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {str(e)}")
            return ""
    
    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generate text using the configured LLM provider.
        
        Parameters:
        -----------
        prompt : str
            The prompt to send to the model
        temperature : float, optional
            Creativity parameter (default: 0.7)
        max_tokens : int, optional
            Maximum response length (default: 1000)
            
        Returns:
        --------
        str
            Generated text
        """
        if not self.is_available:
            logger.warning(f"{self.provider.capitalize()} integration not available.")
            return ""
            
        if self.provider == "gemini":
            return self._generate_with_gemini(prompt, temperature, max_tokens)
        else:
            return self._generate_with_openai(prompt, temperature, max_tokens)
            
    def _prepare_data_description(self, df: pd.DataFrame) -> str:
        """
        Prepare a description of the dataframe for the LLM.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to describe
            
        Returns:
        --------
        str
            A description of the dataframe
        """
        # Basic info
        description = [
            f"This dataset has {df.shape[0]} rows and {df.shape[1]} columns.",
            "The column names and data types are:"
        ]
        
        # Add column info
        for col, dtype in df.dtypes.items():
            description.append(f"- {col}: {dtype}")
            
        # Add sample values for each column
        description.append("\nSample values for each column:")
        
        for col in df.columns:
            sample_values = df[col].dropna().sample(min(3, len(df[col].dropna()))).tolist()
            sample_str = ", ".join([str(val) for val in sample_values])
            description.append(f"- {col}: {sample_str}")
            
        # Add missing value info
        missing = df.isna().sum()
        if missing.sum() > 0:
            description.append("\nMissing values per column:")
            for col, count in missing.items():
                if count > 0:
                    description.append(f"- {col}: {count} missing values ({count/len(df)*100:.1f}%)")
        
        return "\n".join(description)
    
    def _get_statistical_summary(self, df: pd.DataFrame) -> str:
        """
        Generate a statistical summary of the dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to summarize
            
        Returns:
        --------
        str
            A statistical summary
        """
        summary = ["Statistical summary:"]
        
        # Numeric summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary.append("\nNumeric columns summary:")
            summary.append(df[numeric_cols].describe().to_string())
            
        # Categorical summary
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            summary.append("\nTop values for categorical columns:")
            for col in cat_cols:
                if df[col].nunique() < 10:  # Only for columns with few unique values
                    value_counts = df[col].value_counts().head(3)
                    summary.append(f"\n{col}:")
                    for val, count in value_counts.items():
                        summary.append(f"- {val}: {count} ({count/len(df)*100:.1f}%)")
        
        return "\n".join(summary)
    
    def generate_llm_insights(
        self, 
        df: pd.DataFrame, 
        existing_insights: Optional[List[Dict[str, Any]]] = None,
        max_rows: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Generate insights using a language model.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to analyze
        existing_insights : List[Dict[str, Any]], optional
            Existing insights to enhance
        max_rows : int, optional
            Maximum number of rows to include in prompt (default: 1000)
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of insights from the LLM
        """
        if not self.is_available:
            logger.warning(f"{self.provider.capitalize()} package not installed. Skipping LLM insights.")
            return []
            
        # Sample the dataframe if it's too big
        if len(df) > max_rows:
            df_sample = df.sample(max_rows, random_state=42)
        else:
            df_sample = df
            
        # Prepare the prompt
        data_description = self._prepare_data_description(df_sample)
        statistical_summary = self._get_statistical_summary(df_sample)
        
        # Base prompt
        prompt = f"""
You are a data scientist analyzing a dataset. Please provide insights about this data.

{data_description}

{statistical_summary}

Instructions:
1. Analyze the data and provide key insights.
2. Focus on patterns, relationships, and interesting findings.
3. Highlight any anomalies or unusual patterns.
4. Provide 5-7 specific, meaningful insights.
5. For each insight, include:
   - A clear description of the finding
   - Why it might be important
   - A confidence level (high, medium, low)

Format each insight as a single bullet point paragraph.
"""
        
        # Add existing insights if available
        if existing_insights and len(existing_insights) > 0:
            existing_text = "Previously identified insights:\n"
            for insight in existing_insights[:5]:  # Limit to top 5
                existing_text += f"- {insight.get('description', '')}\n"
            prompt += f"\n{existing_text}\nBuild upon these insights with new observations that haven't been mentioned yet."
            
        try:
            # Make the API call
            llm_text = self.generate_text(prompt, temperature=0.7, max_tokens=1000)
            
            if not llm_text:
                return []
            
            # Parse the response into separate insights
            insight_texts = [p for p in llm_text.split('\n') if p.strip() and p.strip().startswith('-')]
            
            # If no bullet points found, try splitting by newlines and numbers
            if not insight_texts:
                lines = llm_text.split('\n')
                insight_texts = []
                current_insight = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        if current_insight:
                            insight_texts.append(' '.join(current_insight))
                            current_insight = []
                    elif line[0].isdigit() and line[1:3] in ['. ', ') ']:
                        if current_insight:
                            insight_texts.append(' '.join(current_insight))
                        current_insight = [line]
                    else:
                        current_insight.append(line)
                        
                if current_insight:
                    insight_texts.append(' '.join(current_insight))
            
            # Convert text insights to structured format
            llm_insights = []
            for i, text in enumerate(insight_texts):
                # Strip bullet points and numbers
                text = text.lstrip('- ').strip()
                if text[:2].isdigit() and text[2:4] in ['. ', ') ']:
                    text = text[4:].strip()
                    
                # Try to extract confidence level
                confidence = 0.7  # Default medium confidence
                if "high confidence" in text.lower():
                    confidence = 0.9
                    text = text.replace("high confidence", "").replace("High confidence", "")
                elif "medium confidence" in text.lower():
                    confidence = 0.7
                    text = text.replace("medium confidence", "").replace("Medium confidence", "")
                elif "low confidence" in text.lower():
                    confidence = 0.5
                    text = text.replace("low confidence", "").replace("Low confidence", "")
                
                llm_insights.append({
                    "type": "llm_insight",
                    "description": text.strip(),
                    "importance": 0.8 - (i * 0.05),  # Decreasing importance by order
                    "confidence": confidence
                })
            
            return llm_insights
            
        except Exception as e:
            logger.error(f"Error generating LLM insights: {str(e)}")
            return []
    
    def generate_data_story(
        self, 
        df: pd.DataFrame,
        insights: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a cohesive data story from insights using LLM.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe
        insights : List[Dict[str, Any]]
            List of insights
            
        Returns:
        --------
        str
            A markdown-formatted data story
        """
        if not self.is_available:
            return ""
            
        # Prepare the prompt
        data_description = self._prepare_data_description(df)
        
        # Convert insights to text
        insight_text = "\n".join([f"- {insight.get('description', '')}" for insight in insights[:10]])
        
        prompt = f"""
You are a data storyteller creating a narrative from data analysis results.

Dataset Information:
{data_description}

Key Insights from Analysis:
{insight_text}

Instructions:
1. Create a comprehensive data story that weaves these insights together.
2. Format the story as a markdown document with clear sections.
3. Include an executive summary at the beginning.
4. Organize insights into logical sections.
5. Add a conclusion section that highlights the most important findings.
6. Keep the tone professional and objective.
7. The story should be 500-800 words.

Your output should be a complete markdown document with headings, bullet points, and paragraphs.
"""
        
        try:
            # Make the API call
            story = self.generate_text(prompt, temperature=0.7, max_tokens=1500)
            return story
            
        except Exception as e:
            logger.error(f"Error generating data story: {str(e)}")
            return ""