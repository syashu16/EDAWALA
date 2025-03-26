"""
Text generation utilities for EDAwala's Storytelling module.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional
import random

class TextGenerator:
    """
    Generates natural language descriptions from data insights.
    """
    
    def __init__(self):
        """Initialize the text generator with templates."""
        # Templates for different insight types
        self.templates = {
            # General templates
            "introduction": [
                "This dataset contains {rows} rows and {columns} columns.",
                "The dataset comprises {rows} records across {columns} features.",
                "Analysis is based on {rows} observations with {columns} variables."
            ],
            
            "summary": [
                "In summary, the key findings from this dataset are: {findings}",
                "The main insights from this analysis are: {findings}",
                "To summarize the analysis: {findings}"
            ],
            
            # Specific insight templates
            "correlation": [
                "There is a {direction} correlation ({correlation:.2f}) between {column1} and {column2}.",
                "{column1} and {column2} show a {direction} relationship (correlation: {correlation:.2f}).",
                "As {column1} increases, {column2} tends to {trend} (correlation: {correlation:.2f})."
            ],
            
            "outliers": [
                "The variable {column} contains {count} outliers ({percent:.1f}% of values).",
                "{count} unusual values ({percent:.1f}%) were detected in {column}.",
                "Outlier analysis revealed {count} extreme values in {column}."
            ],
            
            "missing_values": [
                "Missing data is present in {column}, with {count} values ({percent:.1f}%) missing.",
                "{column} has {percent:.1f}% missing values ({count} total).",
                "Data completeness issue: {percent:.1f}% of {column} values are missing."
            ],
            
            "distribution": [
                "The distribution of {column} is {shape} with a mean of {mean:.2f} and standard deviation of {std:.2f}.",
                "{column} values are {shape} distributed around a mean of {mean:.2f}.",
                "{column} shows a {shape} pattern with values primarily between {q25:.2f} and {q75:.2f}."
            ],
            
            "categorical": [
                "In the {column} category, {dominant_value} is most frequent at {dominant_percent:.1f}%.",
                "The {column} variable is dominated by {dominant_value} ({dominant_percent:.1f}%).",
                "{dominant_value} accounts for {dominant_percent:.1f}% of all {column} values."
            ],
            
            "time_series": [
                "The time series data spans {days} days from {start_date} to {end_date}.",
                "The temporal data covers a period of {days} days, starting at {start_date}.",
                "The dataset contains {days} days of time series data, ending on {end_date}."
            ]
        }
        
        # Templates for section headings
        self.section_headings = {
            "overview": ["Dataset Overview", "Data Summary", "About This Dataset"],
            "missing_data": ["Missing Data Analysis", "Completeness Assessment", "Data Gaps"],
            "distributions": ["Distribution Analysis", "Variable Distributions", "Data Patterns"],
            "relationships": ["Relationship Analysis", "Correlations and Associations", "Feature Interactions"],
            "anomalies": ["Anomaly Detection", "Outliers and Unusual Patterns", "Data Irregularities"],
            "time_analysis": ["Temporal Analysis", "Time Series Insights", "Time-based Patterns"],
            "key_findings": ["Key Findings", "Main Insights", "Important Discoveries"]
        }
    
    def get_random_template(self, template_type: str) -> str:
        """
        Get a random template of the specified type.
        
        Parameters:
        -----------
        template_type : str
            The type of template to retrieve
            
        Returns:
        --------
        str
            A randomly selected template string
        """
        if template_type in self.templates:
            return random.choice(self.templates[template_type])
        else:
            return "{description}"  # Fallback to using the description directly
    
    def get_section_heading(self, section_type: str) -> str:
        """
        Get a heading for a specific section type.
        
        Parameters:
        -----------
        section_type : str
            The type of section
            
        Returns:
        --------
        str
            A heading for the section
        """
        if section_type in self.section_headings:
            return random.choice(self.section_headings[section_type])
        else:
            return section_type.replace('_', ' ').title()  # Fallback to formatted section type
    
    def generate_introduction(self, df: pd.DataFrame) -> str:
        """
        Generate an introduction paragraph for the dataset.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset to describe
            
        Returns:
        --------
        str
            An introduction paragraph
        """
        template = self.get_random_template("introduction")
        
        # Get data types
        dtypes = df.dtypes.value_counts().to_dict()
        dtype_desc = []
        if 'int64' in dtypes or 'float64' in dtypes:
            n_numeric = dtypes.get('int64', 0) + dtypes.get('float64', 0)
            dtype_desc.append(f"{n_numeric} numeric")
        if 'object' in dtypes:
            dtype_desc.append(f"{dtypes['object']} categorical/text")
        if 'datetime64[ns]' in dtypes:
            dtype_desc.append(f"{dtypes['datetime64[ns]']} datetime")
            
        dtype_text = ", ".join(dtype_desc) if dtype_desc else ""
        
        intro = template.format(
            rows=df.shape[0],
            columns=df.shape[1]
        )
        
        if dtype_text:
            intro += f" The data includes {dtype_text} features."
            
        # Add missing values info if present
        missing = df.isna().sum().sum()
        if missing > 0:
            missing_pct = (missing / df.size) * 100
            intro += f" There are {missing} missing values ({missing_pct:.1f}% of the dataset)."
            
        return intro
    
    def format_insight(self, insight: Dict[str, Any]) -> str:
        """
        Format a single insight into a natural language description.
        
        Parameters:
        -----------
        insight : Dict[str, Any]
            The insight dictionary
            
        Returns:
        --------
        str
            Formatted insight text
        """
        # If the insight already has a description, use it
        if "description" in insight:
            return insight["description"]
            
        # Otherwise, try to format using templates
        insight_type = insight.get("type", "")
        
        # Map insight type to template type
        template_type = insight_type
        if "outlier" in insight_type:
            template_type = "outliers"
        elif "missing" in insight_type:
            template_type = "missing_values"
        elif "correlation" in insight_type or "relation" in insight_type:
            template_type = "correlation"
        elif "distribution" in insight_type or "skew" in insight_type:
            template_type = "distribution"
        elif "category" in insight_type:
            template_type = "categorical"
        elif "time" in insight_type:
            template_type = "time_series"
            
        # Get a template
        template = self.get_random_template(template_type)
        
        # Add extra formatting parameters
        if "correlation" in insight_type:
            insight["trend"] = "increase" if insight.get("direction") == "positive" else "decrease"
            
        if "distribution" in insight_type:
            skew = insight.get("skewness", 0)
            if skew > 1:
                insight["shape"] = "right-skewed"
            elif skew < -1:
                insight["shape"] = "left-skewed"
            else:
                insight["shape"] = "relatively symmetric"
                
        # Try to format the template with the insight data
        try:
            return template.format(**insight)
        except KeyError:
            # Fallback to raw description
            return insight.get("description", f"Insight about {insight.get('column', 'data')}")
    
    def generate_story(
        self, 
        df: pd.DataFrame, 
        insights: List[Dict[str, Any]], 
        include_sections: bool = True
    ) -> str:
        """
        Generate a complete data story from insights.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataset
        insights : List[Dict[str, Any]]
            List of insights
        include_sections : bool, optional
            Whether to include section headings (default: True)
            
        Returns:
        --------
        str
            A complete data story with insights organized into sections
        """
        story = []
        
        # Introduction
        if include_sections:
            story.append(f"## {self.get_section_heading('overview')}")
        story.append(self.generate_introduction(df))
        story.append("")  # Empty line
        
        # Organize insights by type
        insight_categories = {
            "missing_data": [],
            "distributions": [],
            "relationships": [],
            "anomalies": [],
            "time_analysis": [],
            "other": []
        }
        
        for insight in insights:
            insight_type = insight.get("type", "")
            
            if "missing" in insight_type:
                insight_categories["missing_data"].append(insight)
            elif "distribution" in insight_type or "skew" in insight_type or "central_tendency" in insight_type:
                insight_categories["distributions"].append(insight)
            elif "correlation" in insight_type or "relation" in insight_type:
                insight_categories["relationships"].append(insight)
            elif "outlier" in insight_type or "anomaly" in insight_type:
                insight_categories["anomalies"].append(insight)
            elif "time" in insight_type:
                insight_categories["time_analysis"].append(insight)
            else:
                insight_categories["other"].append(insight)
        
        # Generate sections
        for category, category_insights in insight_categories.items():
            if not category_insights:
                continue
                
            if include_sections:
                story.append(f"## {self.get_section_heading(category)}")
                
            for insight in category_insights:
                story.append(f"- {self.format_insight(insight)}")
                
            story.append("")  # Empty line
        
        # Key findings
        if include_sections:
            story.append(f"## {self.get_section_heading('key_findings')}")
            
        # Select top insights by importance
        top_insights = sorted(insights, key=lambda x: x.get('importance', 0), reverse=True)[:5]
        finding_texts = [self.format_insight(insight) for insight in top_insights]
        
        template = self.get_random_template("summary")
        summary = template.format(findings=" ".join([f"({i+1}) {text}" for i, text in enumerate(finding_texts)]))
        
        story.append(summary)
        
        return "\n".join(story)