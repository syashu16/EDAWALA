"""
Storytelling EDA (Auto Insights) module for EDAwala.

This module generates natural language insights and narratives from data.
"""

from .insights import generate_insights, generate_story, get_executive_summary, InsightGenerator

__all__ = ['generate_insights', 'generate_story', 'get_executive_summary', 'InsightGenerator']