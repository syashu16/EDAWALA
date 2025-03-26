"""
Statistical insights engine for EDAwala's Storytelling module.
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class StatisticalInsights:
    """
    Extracts statistical insights from dataframes.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The dataframe to analyze
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
    def get_distribution_insights(self) -> List[Dict[str, Any]]:
        """
        Analyze distributions of numeric columns.
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of insights about the distributions
        """
        insights = []
        
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            
            # Skip if not enough data
            if len(col_data) < 10:
                continue
                
            # Calculate basic statistics
            mean = col_data.mean()
            median = col_data.median()
            std = col_data.std()
            skew = col_data.skew()
            # kurt = col_data.kurtosis()  # Not used
            
            # Detect skewness
            if abs(skew) > 1:
                direction = "right" if skew > 0 else "left"
                insights.append({
                    "type": "skewness",
                    "column": col,
                    "skewness": skew,
                    "direction": direction,
                    "importance": min(abs(skew) / 2, 1) * 0.8,  # Scale importance
                    "description": f"The distribution of {col} is skewed to the {direction} (skewness = {skew:.2f})."
                })
            
            # Detect if mean and median are significantly different
            if abs(mean - median) > 0.1 * std and std > 0:
                insights.append({
                    "type": "central_tendency",
                    "column": col,
                    "mean": mean,
                    "median": median,
                    "difference": abs(mean - median),
                    "importance": min(abs(mean - median) / std, 1) * 0.7,
                    "description": f"The mean ({mean:.2f}) and median ({median:.2f}) of {col} differ significantly, suggesting the presence of outliers or skewed distribution."
                })
            
            # Detect bimodality or multimodality using kernel density estimation
            try:
                from scipy.signal import find_peaks
                
                # Create a kernel density estimate
                kde = stats.gaussian_kde(col_data)
                x = np.linspace(col_data.min(), col_data.max(), 1000)
                y = kde(x)
                
                # Find peaks
                peaks, _ = find_peaks(y, height=0.1*np.max(y))
                
                if len(peaks) > 1:
                    insights.append({
                        "type": "multimodality",
                        "column": col,
                        "n_peaks": len(peaks),
                        "importance": 0.85,
                        "description": f"The distribution of {col} appears to have {len(peaks)} peaks, suggesting multiple modes or clusters."
                    })
            except:
                # Skip if KDE fails
                pass
        
        return insights
    
    def get_correlation_insights(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find strong correlations between numeric columns.
        
        Parameters:
        -----------
        threshold : float, optional
            Correlation strength threshold (default: 0.7)
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of insights about correlations
        """
        insights = []
        
        # Need at least 2 numeric columns
        if len(self.numeric_cols) < 2:
            return insights
            
        # Calculate correlation matrix
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find strong correlations
        for i in range(len(self.numeric_cols)):
            for j in range(i+1, len(self.numeric_cols)):
                col1 = self.numeric_cols[i]
                col2 = self.numeric_cols[j]
                corr = corr_matrix.loc[col1, col2]
                
                if abs(corr) >= threshold:
                    direction = "positive" if corr > 0 else "negative"
                    strength = "strong" if abs(corr) > 0.8 else "moderate"
                    
                    insights.append({
                        "type": "correlation",
                        "column1": col1,
                        "column2": col2,
                        "correlation": corr,
                        "direction": direction,
                        "importance": abs(corr),
                        "description": f"There is a {strength} {direction} correlation ({corr:.2f}) between {col1} and {col2}."
                    })
        
        return insights
    
    def get_outlier_insights(self, z_threshold: float = 3.0) -> List[Dict[str, Any]]:
        """
        Detect outliers in numeric columns.
        
        Parameters:
        -----------
        z_threshold : float, optional
            Z-score threshold for outliers (default: 3.0)
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of insights about outliers
        """
        insights = []
        
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            
            # Skip if not enough data
            if len(col_data) < 10:
                continue
                
            # Calculate z-scores
            z_scores = np.abs(stats.zscore(col_data))
            
            # Count outliers
            outliers = (z_scores > z_threshold).sum()
            
            if outliers > 0:
                pct_outliers = (outliers / len(col_data)) * 100
                
                insights.append({
                    "type": "outliers",
                    "column": col,
                    "count": int(outliers),
                    "percent": pct_outliers,
                    "importance": min((pct_outliers / 10) + 0.5, 1.0),
                    "description": f"Found {outliers} outliers ({pct_outliers:.1f}%) in {col} based on z-score > {z_threshold}."
                })
        
        return insights
    
    def get_missing_value_insights(self, threshold: float = 0.05) -> List[Dict[str, Any]]:
        """
        Generate insights about missing values.
        
        Parameters:
        -----------
        threshold : float, optional
            Minimum percentage of missing values to report (default: 0.05)
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of insights about missing values
        """
        insights = []
        
        # Calculate missing values
        missing = self.df.isna().sum()
        missing_pct = missing / len(self.df)
        
        # Overall missing values
        total_missing = missing.sum()
        total_missing_pct = total_missing / self.df.size
        
        if total_missing > 0:
            insights.append({
                "type": "missing_overall",
                "count": int(total_missing),
                "percent": total_missing_pct * 100,
                "importance": min(total_missing_pct * 2, 0.9),
                "description": f"Overall, {total_missing} values ({total_missing_pct*100:.1f}%) are missing in the dataset."
            })
        
        # Column-specific missing values
        for col in self.df.columns:
            if missing_pct[col] >= threshold:
                insights.append({
                    "type": "missing_column",
                    "column": col,
                    "count": int(missing[col]),
                    "percent": missing_pct[col] * 100,
                    "importance": min(missing_pct[col] + 0.3, 0.95),
                    "description": f"Column '{col}' has {missing[col]} missing values ({missing_pct[col]*100:.1f}%)."
                })
        
        return insights
    
    def get_categorical_insights(self) -> List[Dict[str, Any]]:
        """
        Generate insights about categorical columns.
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of insights about categorical variables
        """
        insights = []
        
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            unique_count = len(value_counts)
            
            # Skip if too many unique values (likely not categorical)
            if unique_count > 20:
                continue
                
            # Check for imbalanced categories
            total_count = value_counts.sum()
            top_category = value_counts.index[0]
            top_count = value_counts.iloc[0]
            top_pct = (top_count / total_count) * 100
            
            if top_pct > 75:
                insights.append({
                    "type": "imbalanced_category",
                    "column": col,
                    "dominant_value": top_category,
                    "dominant_percent": top_pct,
                    "importance": min((top_pct - 75) / 25 + 0.6, 0.9),
                    "description": f"The category '{col}' is highly imbalanced with '{top_category}' accounting for {top_pct:.1f}% of all values."
                })
            
            # Check for high cardinality
            if 10 <= unique_count <= 20:
                insights.append({
                    "type": "high_cardinality",
                    "column": col,
                    "unique_count": unique_count,
                    "importance": 0.6,
                    "description": f"Column '{col}' has high cardinality with {unique_count} unique values."
                })
        
        return insights
    
    def get_time_series_insights(self) -> List[Dict[str, Any]]:
        """
        Generate insights for time series data if present.
        
        Returns:
        --------
        List[Dict[str, Any]]
            List of insights about time series data
        """
        insights = []
        
        # Skip if no datetime columns
        if not self.datetime_cols:
            # Try to infer datetime columns
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    try:
                        pd.to_datetime(self.df[col], errors='raise')
                        self.datetime_cols.append(col)
                    except:
                        pass
                        
        if not self.datetime_cols:
            return insights
            
        # Analyze each datetime column
        for date_col in self.datetime_cols:
            # Convert to datetime if not already
            if self.df[date_col].dtype != 'datetime64[ns]':
                try:
                    date_series = pd.to_datetime(self.df[date_col])
                except:
                    continue
            else:
                date_series = self.df[date_col]
                
            # Skip if not enough data
            if len(date_series.dropna()) < 10:
                continue
                
            # Date range
            min_date = date_series.min()
            max_date = date_series.max()
            date_range = max_date - min_date
            
            insights.append({
                "type": "time_range",
                "column": date_col,
                "start_date": min_date,
                "end_date": max_date,
                "days": date_range.days,
                "importance": 0.7,
                "description": f"The data spans {date_range.days} days from {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}."
            })
            
            # Check for gaps
            if len(date_series.dropna()) >= 30:
                try:
                    # Sort dates
                    sorted_dates = date_series.dropna().sort_values()
                    
                    # Calculate differences between consecutive dates
                    date_diffs = sorted_dates.diff().dropna()
                    
                    # Find the median difference
                    median_diff = date_diffs.median()
                    
                    # Find gaps (differences > 2x median)
                    gaps = date_diffs[date_diffs > 2 * median_diff]
                    
                    if len(gaps) > 0:
                        insights.append({
                            "type": "time_gaps",
                            "column": date_col,
                            "gap_count": len(gaps),
                            "importance": min(0.6 + (len(gaps) / 100), 0.9),
                            "description": f"Found {len(gaps)} significant gaps in the time series data for '{date_col}'."
                        })
                except:
                    pass
        
        return insights
    
    def get_all_insights(self) -> List[Dict[str, Any]]:
        """
        Get all available statistical insights.
        
        Returns:
        --------
        List[Dict[str, Any]]
            Combined list of all insights
        """
        all_insights = []
        
        # Collect insights from all methods
        all_insights.extend(self.get_distribution_insights())
        all_insights.extend(self.get_correlation_insights())
        all_insights.extend(self.get_outlier_insights())
        all_insights.extend(self.get_missing_value_insights())
        all_insights.extend(self.get_categorical_insights())
        all_insights.extend(self.get_time_series_insights())
        
        # Sort by importance
        all_insights.sort(key=lambda x: x.get('importance', 0), reverse=True)
        
        return all_insights