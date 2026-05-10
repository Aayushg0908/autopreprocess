"""
Quality Detector - Analyzes Data Quality Issues

Detects:
1. Missing values (nulls)
2. Outliers (anomalies using IQR method)
3. Skewness (distribution asymmetry)
4. Correlations (multicollinearity between features)

Why it's important:
- Missing values need handling before training
- Outliers can skew models like Linear Regression
- Skewed data may need log transformation
- High correlations cause multicollinearity issues
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

class QualityDetector:
    """
    Analyzes data quality issues in a DataFrame
    
    Usage:
        detector = QualityDetector()
        missing_report = detector.detect_missing_values(df)
        outlier_report = detector.detect_outliers_iqr(df)
        skewness_report = detector.detect_skewness(df)
        correlation_report = detector.detect_correlations(df)
    """
    
    @staticmethod
    def detect_missing_values(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Detect missing values in each column
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary with column name → missing stats
        
        Example:
            >>> detector.detect_missing_values(df)
            {
                'age': {'count': 0, 'percentage': 0.0, 'recommendation': 'none'},
                'income': {'count': 150, 'percentage': 15.0, 'recommendation': 'impute'},
                'city': {'count': 500, 'percentage': 50.0, 'recommendation': 'drop'}
            }
        """
        missing_report = {}
        total_rows = len(df)
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / total_rows) * 100
            
            # Generate recommendation based on missing percentage
            if missing_pct == 0:
                recommendation = 'none'
            elif missing_pct < 5:
                recommendation = 'impute_simple'  # Fill with median/mode
            elif missing_pct < 20:
                recommendation = 'impute_advanced'  # KNN or MICE
            elif missing_pct < 40:
                recommendation = 'impute_with_indicator'  # Add missing flag column
            else:
                recommendation = 'drop_column'  # Too many missing
            
            missing_report[col] = {
                'count': missing_count,
                'percentage': round(missing_pct, 2),
                'recommendation': recommendation
            }
        
        return missing_report
    
    @staticmethod
    def detect_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers using IQR (Interquartile Range) method
        
        How IQR works:
        1. Sort data and find Q1 (25th percentile) and Q3 (75th percentile)
        2. IQR = Q3 - Q1 (middle 50% of data)
        3. Outliers are values below Q1 - multiplier*IQR or above Q3 + multiplier*IQR
        
        Args:
            df: Input DataFrame
            multiplier: Usually 1.5 (moderate) or 3 (extreme outliers)
        
        Returns:
            Dictionary with outlier stats for each numeric column
        
        Example:
            >>> detector.detect_outliers_iqr(df)
            {
                'income': {'count': 12, 'percentage': 1.2, 'lower_bound': 20000, 'upper_bound': 200000},
                'age': {'count': 0, 'percentage': 0.0}
            }
        """
        outlier_report = {}
        total_rows = len(df)
        
        # Only check numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Drop missing values for calculation
            clean_data = df[col].dropna()
            
            if len(clean_data) == 0:
                continue
            
            # Calculate quartiles
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Calculate bounds
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # Find outliers
            outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / total_rows) * 100
            
            outlier_report[col] = {
                'count': outlier_count,
                'percentage': round(outlier_pct, 2),
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'has_outliers': outlier_count > 0
            }
        
        return outlier_report
    
    @staticmethod
    def detect_skewness(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Dict[str, Any]]:
        """
        Detect skewness in numeric columns
        
        Skewness interpretation:
        - 0 = perfectly symmetric (normal distribution)
        - 0.5 to 1 = moderate right skew
        - > 1 = high right skew (log transform recommended)
        - -0.5 to -1 = moderate left skew
        - < -1 = high left skew
        
        Args:
            df: Input DataFrame
            threshold: Threshold for considering data skewed (default 0.5)
        
        Returns:
            Dictionary with skewness stats for each numeric column
        
        Example:
            >>> detector.detect_skewness(df)
            {
                'income': {'skewness': 2.3, 'recommendation': 'log_transform'},
                'age': {'skewness': 0.2, 'recommendation': 'none'}
            }
        """
        skewness_report = {}
        
        # Only check numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Drop missing values
            clean_data = df[col].dropna()
            
            if len(clean_data) == 0:
                continue
            
            # Calculate skewness
            skew_value = clean_data.skew()
            
            # Generate recommendation
            if abs(skew_value) < threshold:
                recommendation = 'none'
            elif skew_value > threshold:
                if skew_value > 1:
                    recommendation = 'log_transform_strong'
                else:
                    recommendation = 'log_transform_mild'
            else:  # skew_value < -threshold
                if skew_value < -1:
                    recommendation = 'square_transform'
                else:
                    recommendation = 'none'
            
            skewness_report[col] = {
                'skewness': round(skew_value, 3),
                'recommendation': recommendation,
                'is_skewed': abs(skew_value) > threshold
            }
        
        return skewness_report
    
    @staticmethod
    def detect_correlations(df: pd.DataFrame, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Detect highly correlated feature pairs (multicollinearity)
        
        Args:
            df: Input DataFrame
            threshold: Correlation threshold (default 0.8)
        
        Returns:
            List of tuples: (feature1, feature2, correlation_value)
        
        Example:
            >>> detector.detect_correlations(df)
            [
                ('total_charges', 'tenure', 0.89),
                ('income', 'education_years', 0.82)
            ]
        """
        # Only check numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return []  # Not enough numeric columns for correlation
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        
        # Loop through upper triangle of correlation matrix
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        round(corr_value, 3)
                    ))
        
        # Sort by correlation strength (descending)
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return high_corr_pairs
    
    @staticmethod
    def get_full_quality_report(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
        """
        Generate a complete quality report combining all detectors
        
        Args:
            df: Input DataFrame
            verbose: Print summary
        
        Returns:
            Dictionary with all quality metrics
        """
        if verbose:
            print("\n" + "="*60)
            print("📊 DATA QUALITY REPORT")
            print("="*60)
        
        # Run all detectors
        missing_report = QualityDetector.detect_missing_values(df)
        outlier_report = QualityDetector.detect_outliers_iqr(df)
        skewness_report = QualityDetector.detect_skewness(df)
        correlation_pairs = QualityDetector.detect_correlations(df)
        
        # Calculate quality score (0-100)
        quality_score = 100
        
        # Penalize for missing values
        for col, stats in missing_report.items():
            if stats['percentage'] > 20:
                quality_score -= 5
            elif stats['percentage'] > 5:
                quality_score -= 2
        
        # Penalize for outliers
        for col, stats in outlier_report.items():
            if stats.get('has_outliers', False):
                if stats['percentage'] > 5:
                    quality_score -= 3
                elif stats['percentage'] > 1:
                    quality_score -= 1
        
        # Penalize for high skewness
        for col, stats in skewness_report.items():
            if stats.get('is_skewed', False):
                if abs(stats['skewness']) > 1:
                    quality_score -= 2
        
        # Penalize for high correlations
        if len(correlation_pairs) > 3:
            quality_score -= 5
        elif len(correlation_pairs) > 0:
            quality_score -= 2
        
        quality_score = max(0, min(100, quality_score))
        
        if verbose:
            print(f"\n📈 Overall Quality Score: {quality_score}/100")
            
            # Summary of issues
            total_missing = sum(stats['count'] for stats in missing_report.values())
            total_outliers = sum(stats['count'] for stats in outlier_report.values())
            skewed_cols = [col for col, stats in skewness_report.items() if stats['is_skewed']]
            
            print(f"\n📋 Issues Summary:")
            print(f"   Missing values: {total_missing} total")
            print(f"   Outliers: {total_outliers} total")
            print(f"   Skewed columns: {len(skewed_cols)}")
            print(f"   High correlations: {len(correlation_pairs)} pairs")
            print("="*60)
        
        return {
            'missing': missing_report,
            'outliers': outlier_report,
            'skewness': skewness_report,
            'correlations': correlation_pairs,
            'quality_score': quality_score
        }