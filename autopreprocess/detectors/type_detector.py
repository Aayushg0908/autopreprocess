"""
Type Detector - Automatically Identifies Column Types

Updated version with more nuanced classification:
- continuous: Many unique numeric values (age, price)
- discrete_numeric: Few unique numbers, order matters (rating 1-5)
- categorical: No inherent order (city, color)
- datetime: Date/time values
- useless: No predictive value (IDs, constants)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

class TypeDetector:
    
    @staticmethod
    def detect_column_type(series: pd.Series, continuous_threshold: int = 20) -> str:
        """
        Detect column type with more nuance
        
        Returns:
            'continuous' - Many unique numeric values (scale me!)
            'discrete_numeric' - Few unique numbers, order matters (keep as numbers)
            'categorical' - No natural order (one-hot encode me!)
            'datetime' - Date/time values
            'useless' - Drop me!
        """
        
        col_name = series.name if hasattr(series, 'name') else 'unknown'
        clean_series = series.dropna()
        
        # CASE 1: All missing → useless
        if len(clean_series) == 0:
            print(f"  ⚠️ Column '{col_name}': All missing → useless")
            return 'useless'
        
        # CASE 2: Constant column → useless
        if clean_series.nunique() == 1:
            print(f"  ℹ️ Column '{col_name}': Constant → useless")
            return 'useless'
        
        # CASE 3: Identifier detection (all unique values)
        if clean_series.nunique() == len(clean_series) and len(clean_series) > 10:
            if pd.api.types.is_numeric_dtype(series):
                sorted_values = clean_series.sort_values().values
                is_sequential = all(sorted_values[i+1] - sorted_values[i] == 1 
                                   for i in range(len(sorted_values)-1))
                if is_sequential:
                    print(f"  ℹ️ Column '{col_name}': Sequential ID → useless")
                    return 'useless'
        
        # CASE 4: Numeric columns
        if pd.api.types.is_numeric_dtype(series):
            unique_count = clean_series.nunique()
            
            # Check if values are likely ordinal (evenly spaced, small range)
            min_val = clean_series.min()
            max_val = clean_series.max()
            range_val = max_val - min_val
            
            # If few unique values AND they're small integers (like 1-5)
            if unique_count <= 10 and range_val <= 10 and min_val >= 0:
                print(f"  ✅ Column '{col_name}': {unique_count} unique numeric values (range {min_val}-{max_val}) → discrete_numeric")
                return 'discrete_numeric'
            
            # Many unique values → continuous numeric
            elif unique_count > continuous_threshold:
                print(f"  ✅ Column '{col_name}': {unique_count} unique values → continuous")
                return 'continuous'
            
            # Medium unique values but large range → could go either way
            else:
                print(f"  ✅ Column '{col_name}': {unique_count} unique values, range {min_val}-{max_val} → continuous (for scaling)")
                return 'continuous'
        
        # CASE 5: Datetime (only if not numeric)
        if TypeDetector._is_datetime(clean_series):
            print(f"  ✅ Column '{col_name}': Detected as datetime")
            return 'datetime'
        
        # CASE 6: Default to categorical
        print(f"  ✅ Column '{col_name}': Detected as categorical")
        return 'categorical'
    
    @staticmethod
    def _is_datetime(series: pd.Series) -> bool:
        """Check if series contains datetime values"""
        sample = series.head(10)
        try:
            pd.to_datetime(sample, errors='raise')
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def analyze_dataframe(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
        """Analyze all columns with new classification"""
        
        if verbose:
            print(f"\n🔍 Analyzing {len(df.columns)} columns...")
            print("="*50)
        
        analysis = {
            'continuous': [],          # Many unique numbers (scale)
            'discrete_numeric': [],    # Few unique numbers (keep as numbers)
            'categorical': [],         # No order (one-hot)
            'datetime': [],            # Dates
            'useless': [],             # Drop
            'stats': {}
        }
        
        for col in df.columns:
            col_type = TypeDetector.detect_column_type(df[col])
            analysis[col_type].append(col)
            
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            
            analysis['stats'][col] = {
                'dtype': str(df[col].dtype),
                'null_count': null_count,
                'null_pct': null_pct,
                'unique_count': unique_count,
                'type': col_type
            }
        
        if verbose:
            print("="*50)
            print(f"📊 Analysis Complete:")
            print(f"   Continuous (scale): {len(analysis['continuous'])}")
            print(f"   Discrete numeric: {len(analysis['discrete_numeric'])}")
            print(f"   Categorical (encode): {len(analysis['categorical'])}")
            print(f"   Datetime: {len(analysis['datetime'])}")
            print(f"   Useless (drop): {len(analysis['useless'])}")
            print("="*50)
        
        return analysis
    
    @staticmethod
    def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
        """Return summary DataFrame"""
        analysis = TypeDetector.analyze_dataframe(df, verbose=False)
        
        summary_data = []
        for col in df.columns:
            stats = analysis['stats'][col]
            summary_data.append({
                'Column': col,
                'Detected_Type': stats['type'],
                'Null_%': stats['null_pct'],
                'Unique_Count': stats['unique_count']
            })
        
        return pd.DataFrame(summary_data)