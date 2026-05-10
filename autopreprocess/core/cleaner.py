"""
Cleaner - Handles Missing Values, Outliers, Duplicates, and Useless Columns

What it does:
1. Fills or removes missing values intelligently
2. Caps outliers at percentile boundaries
3. Removes duplicate rows
4. Drops columns that have no predictive value

Why it's important:
- Missing values break most ML algorithms
- Outliers can skew models like Linear Regression
- Duplicates waste memory and cause overfitting
- Useless columns add noise and slow down training
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

class DataCleaner:
    """
    Cleans a DataFrame by handling common data quality issues
    
    Usage:
        cleaner = DataCleaner()
        df_clean = cleaner.clean_dataframe(df)
    """
    
    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        missing_threshold_drop: float = 40.0,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Handle missing values in DataFrame
        
        Strategy based on missing percentage:
        - < 5%: Fill with median (numeric) or mode (categorical)
        - 5-20%: Fill with KNN imputation (simple version uses mean/mode)
        - 20-40%: Create missing indicator column + fill
        - > 40%: Drop column
        
        Args:
            df: Input DataFrame
            missing_threshold_drop: Drop column if missing % above this (default 40)
            verbose: Print actions taken
        
        Returns:
            Tuple of (cleaned_df, actions_taken)
        """
        
        df_clean = df.copy()
        actions = {}
        total_rows = len(df_clean)
        
        if verbose:
            print("\n🔧 HANDLING MISSING VALUES")
            print("="*40)
        
        for col in df_clean.columns:
            missing_count = df_clean[col].isnull().sum()
            missing_pct = (missing_count / total_rows) * 100
            
            if missing_count == 0:
                continue  # No missing values, skip
            
            # Determine strategy based on missing percentage
            if missing_pct >= missing_threshold_drop:
                # Drop column
                df_clean = df_clean.drop(columns=[col])
                actions[col] = f"dropped ({missing_pct:.1f}% missing)"
                if verbose:
                    print(f"  ❌ Dropped '{col}': {missing_pct:.1f}% missing")
            
            elif missing_pct >= 20:
                # Create missing indicator column + fill
                indicator_col = f"{col}_was_missing"
                df_clean[indicator_col] = df_clean[col].isnull().astype(int)
                
                # Fill with median/mode
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_value = df_clean[col].median()
                else:
                    fill_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else "Unknown"
                
                df_clean[col] = df_clean[col].fillna(fill_value)
                actions[col] = f"filled with {fill_value}, added '{indicator_col}' flag ({missing_pct:.1f}% missing)"
                if verbose:
                    print(f"  📝 '{col}': {missing_pct:.1f}% missing → filled + added indicator")
            
            elif missing_pct >= 5:
                # Advanced imputation (simplified - uses median/mode for now)
                # In production, you'd use KNN or MICE here
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_value = df_clean[col].median()
                else:
                    fill_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else "Unknown"
                
                df_clean[col] = df_clean[col].fillna(fill_value)
                actions[col] = f"imputed with {fill_value} ({missing_pct:.1f}% missing)"
                if verbose:
                    print(f"  🔧 '{col}': {missing_pct:.1f}% missing → imputed with {fill_value}")
            
            else:
                # Simple imputation (<5% missing)
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_value = df_clean[col].median()
                else:
                    fill_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else "Unknown"
                
                df_clean[col] = df_clean[col].fillna(fill_value)
                actions[col] = f"filled with {fill_value} ({missing_pct:.1f}% missing)"
                if verbose:
                    print(f"  ✅ '{col}': {missing_pct:.1f}% missing → filled with {fill_value}")
        
        if verbose and len(actions) == 0:
            print("  ✅ No missing values found!")
        
        return df_clean, actions
    
    @staticmethod
    def handle_outliers(
        df: pd.DataFrame,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """
        Cap outliers at specified percentiles
        
        How it works:
        1. Find P1 and P99 (or custom percentiles)
        2. Any value < P1 becomes P1
        3. Any value > P99 becomes P99
        
        This is called "Winsorization" - it keeps outliers but limits their impact.
        
        Args:
            df: Input DataFrame
            lower_percentile: Lower bound percentile (default 1)
            upper_percentile: Upper bound percentile (default 99)
            verbose: Print actions taken
        
        Returns:
            Tuple of (cleaned_df, outlier_info)
        """
        
        df_clean = df.copy()
        outlier_info = {}
        
        # Only process numeric columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if verbose:
            print("\n🔧 HANDLING OUTLIERS")
            print("="*40)
        
        for col in numeric_cols:
            # Skip if all values are same (constant)
            if df_clean[col].nunique() == 1:
                continue
            
            # Calculate percentiles
            lower_bound = df_clean[col].quantile(lower_percentile / 100)
            upper_bound = df_clean[col].quantile(upper_percentile / 100)
            
            # Count outliers before capping
            outliers_before = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            outlier_pct = (outliers_before / len(df_clean)) * 100
            
            if outliers_before == 0:
                continue
            
            # Cap outliers
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            
            outlier_info[col] = {
                'lower_bound': round(lower_bound, 2),
                'upper_bound': round(upper_bound, 2),
                'outliers_capped': int(outliers_before),
                'outlier_pct': round(outlier_pct, 2)
            }
            
            if verbose:
                print(f"  ✂️ '{col}': capped {outliers_before} outliers ({outlier_pct:.1f}%)")
                print(f"     Range limited to [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        if verbose and len(outlier_info) == 0:
            print("  ✅ No outliers found!")
        
        return df_clean, outlier_info
    
    @staticmethod
    def remove_duplicates(
        df: pd.DataFrame,
        keep: str = 'first',
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, int]:
        """
        Remove duplicate rows from DataFrame
        
        Args:
            df: Input DataFrame
            keep: Which duplicate to keep ('first', 'last', or False for none)
            verbose: Print actions taken
        
        Returns:
            Tuple of (cleaned_df, number_of_duplicates_removed)
        """
        
        duplicates_before = df.duplicated().sum()
        
        if duplicates_before == 0:
            if verbose:
                print("\n🔧 REMOVING DUPLICATES")
                print("="*40)
                print("  ✅ No duplicate rows found!")
            return df.copy(), 0
        
        df_clean = df.drop_duplicates(keep=keep)
        duplicates_removed = duplicates_before
        
        if verbose:
            print("\n🔧 REMOVING DUPLICATES")
            print("="*40)
            print(f"  🗑️ Removed {duplicates_removed} duplicate rows")
            print(f"  📊 Shape: {df.shape} → {df_clean.shape}")
        
        return df_clean, duplicates_removed
    
    @staticmethod
    def drop_useless_columns(
        df: pd.DataFrame,
        verbose: bool = True
    ) -> tuple:
        """
        Drop columns that are useless (constant or sequential identifiers)
        
        Args:
            df: Input DataFrame
            verbose: Print actions taken
        
        Returns:
            Tuple of (cleaned_df, list_of_dropped_columns)
        """
        import pandas as pd
        
        df_clean = df.copy()
        dropped = []
        
        if verbose:
            print("\n🔧 DROPPING USELESS COLUMNS")
            print("="*40)
        
        for col in df_clean.columns:
            # Check if constant (all same value)
            if df_clean[col].nunique() == 1:
                df_clean = df_clean.drop(columns=[col])
                dropped.append(col)
                if verbose:
                    constant_value = str(df[col].iloc[0]) if len(df) > 0 else "N/A"
                    if len(constant_value) > 30:
                        constant_value = constant_value[:27] + "..."
                    print(f"  🗑️ Dropped '{col}': constant value '{constant_value}'")
            
            # Check if identifier (sequential numbers, all unique)
            elif df_clean[col].nunique() == len(df_clean) and len(df_clean) > 10:
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    sorted_vals = df_clean[col].dropna().sort_values().values
                    if len(sorted_vals) > 1:
                        # Check if values are sequential (difference of 1)
                        is_sequential = all(
                            sorted_vals[i+1] - sorted_vals[i] == 1 
                            for i in range(len(sorted_vals)-1)
                        )
                        if is_sequential:
                            df_clean = df_clean.drop(columns=[col])
                            dropped.append(col)
                            if verbose:
                                print(f"  🗑️ Dropped '{col}': sequential ID")
        
        if verbose and len(dropped) == 0:
            print("  ✅ No useless columns found")
        elif verbose and len(dropped) > 0:
            print(f"  📊 Total useless columns dropped: {len(dropped)}")
        
        return df_clean, dropped
    @staticmethod
    def clean_dataframe(
        df: pd.DataFrame,
        handle_missing: bool = True,
        handle_outliers: bool = True,
        remove_duplicates: bool = True,
        drop_useless: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Main cleaning function - runs all cleaning steps
        
        Args:
            df: Input DataFrame
            handle_missing: Whether to handle missing values
            handle_outliers: Whether to handle outliers
            remove_duplicates: Whether to remove duplicate rows
            drop_useless: Whether to drop useless columns
            verbose: Print progress
        
        Returns:
            Dictionary with cleaned DataFrame and cleaning report
        """
        
        df_clean = df.copy()
        cleaning_report = {
            'original_shape': df.shape,
            'steps_performed': [],
            'missing_handling': {},
            'outlier_handling': {},
            'duplicates_removed': 0,
            'useless_columns_dropped': [],
            'final_shape': None
        }
        
        if verbose:
            print("\n" + "="*60)
            print("🧹 DATA CLEANING PIPELINE")
            print("="*60)
            print(f"📊 Original shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Step 1: Remove duplicates
        if remove_duplicates:
            df_clean, dup_removed = DataCleaner.remove_duplicates(df_clean, verbose=verbose)
            cleaning_report['duplicates_removed'] = dup_removed
            if dup_removed > 0:
                cleaning_report['steps_performed'].append('removed_duplicates')
        
        # Step 2: Drop useless columns
        if drop_useless:
            df_clean, dropped = DataCleaner.drop_useless_columns(df_clean, verbose=verbose)
            cleaning_report['useless_columns_dropped'] = dropped
            if dropped:
                cleaning_report['steps_performed'].append('dropped_useless_columns')
        
        # Step 3: Handle missing values
        if handle_missing:
            df_clean, missing_actions = DataCleaner.handle_missing_values(df_clean, verbose=verbose)
            cleaning_report['missing_handling'] = missing_actions
            if missing_actions:
                cleaning_report['steps_performed'].append('handled_missing_values')
        
        # Step 4: Handle outliers
        if handle_outliers:
            df_clean, outlier_actions = DataCleaner.handle_outliers(df_clean, verbose=verbose)
            cleaning_report['outlier_handling'] = outlier_actions
            if outlier_actions:
                cleaning_report['steps_performed'].append('handled_outliers')
        
        cleaning_report['final_shape'] = df_clean.shape
        
        if verbose:
            print("\n" + "="*40)
            print("✅ CLEANING COMPLETE")
            print("="*40)
            print(f"📊 Final shape: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
            print(f"📈 Rows removed: {df.shape[0] - df_clean.shape[0]}")
            print(f"📉 Columns removed: {df.shape[1] - df_clean.shape[1]}")
        
        cleaning_report['cleaned_data'] = df_clean
        
        return cleaning_report