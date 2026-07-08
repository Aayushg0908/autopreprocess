"""
Scaler - Scale numeric features using StandardScaler, MinMaxScaler, or RobustScaler

Key principle: fit() on TRAIN only, transform() on both train and test
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any

class Scaler:
    """
    Scale numeric columns using various methods.
    
    Usage:
        scaler = Scaler(method='standard', verbose=True)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    Methods:
        'standard' - (x - mean) / std  (default)
        'minmax'   - (x - min) / (max - min) -> range [0,1]
        'robust'   - (x - median) / IQR   (handles outliers)
    """
    
    def __init__(self, 
                 numeric_cols: Optional[List[str]] = None,
                 method: str = 'standard',
                 verbose: bool = True):
        """
        Initialize Scaler.
        
        Args:
            numeric_cols: Columns to scale (auto-detect if None)
            method: 'standard', 'minmax', or 'robust'
            verbose: Print progress
        """
        self.numeric_cols = numeric_cols
        self.method = method
        self.verbose = verbose
        
        # Store fitted parameters
        self.params_ = {}  # {col: {'mean': x, 'std': y}} etc.
    
    def _detect_numeric_cols(self, df: pd.DataFrame) -> List[str]:
        from ..detectors.type_detector import TypeDetector
        detector = TypeDetector()
        analysis = detector.analyze_dataframe(df, verbose=False)
        
        # Exclude binary columns (0/1 values from one-hot encoding)
        continuous = []
        for col in analysis['continuous']:
            unique_count = df[col].nunique()
            # Skip columns that look like binary encoded (only 0 and 1)
            if unique_count == 2 and set(df[col].dropna().unique()).issubset({0, 1}):
                if self.verbose:
                    print(f"  ℹ️ Skipping binary column '{col}' (already scaled 0-1)")
                continue
            continuous.append(col)
        
        return continuous
    
    def fit(self, X: pd.DataFrame) -> 'Scaler':
        """
        Calculate scaling parameters from TRAINING data only.
        
        Args:
            X: Training data
        
        Returns:
            self (fitted scaler)
        """
        if self.verbose:
            print(f"\n🔧 FITTING SCALER ({self.method})")
            print("="*40)
        
        # Detect numeric columns if not specified
        if self.numeric_cols is None:
            self.numeric_cols = self._detect_numeric_cols(X)
            if self.verbose:
                print(f"   Detected numeric columns: {self.numeric_cols}")
        if not self.numeric_cols:
            if self.verbose:
                print("  ℹ️ No numeric columns to scale")
            return self
        # Calculate parameters for each numeric column
        for col in self.numeric_cols:
            if col not in X.columns:
                continue
            
            # Drop NaN for calculation
            clean_data = X[col].dropna()
            
            if self.method == 'standard':
                mean = clean_data.mean()
                std = clean_data.std()
                # Handle zero standard deviation (all values identical)
                if std == 0:
                    std = 1.0  # Avoid division by zero
                    if self.verbose:
                        print(f"  ⚠️ '{col}': std=0, set to 1 (constant column)")
                self.params_[col] = {'mean': mean, 'std': std}
                if self.verbose:
                    print(f"  ✅ '{col}': mean={mean:.2f}, std={std:.2f}")
            
            elif self.method == 'minmax':
                # MinMaxScaler: (x - min) / (max - min)
                min_val = clean_data.min()
                max_val = clean_data.max()
                self.params_[col] = {'min': min_val, 'max': max_val}
                if self.verbose:
                    print(f"  ✅ '{col}': min={min_val:.2f}, max={max_val:.2f}")
            
            elif self.method == 'robust':
                # RobustScaler: (x - median) / IQR
                median = clean_data.median()
                q1 = clean_data.quantile(0.25)
                q3 = clean_data.quantile(0.75)
                iqr = q3 - q1
                self.params_[col] = {'median': median, 'iqr': iqr}
                if self.verbose:
                    print(f"  ✅ '{col}': median={median:.2f}, IQR={iqr:.2f}")
        
        if self.verbose:
            print("="*40)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling using parameters from fit().
        
        Args:
            X: Data to transform (train or test)
        
        Returns:
            Scaled DataFrame
        """
        if not self.params_:
            if self.verbose:
                print("  ℹ️ No numeric columns to transform - returning original data")
            return X.copy()
        
        if self.verbose:
            print(f"\n🔄 TRANSFORMING DATA ({self.method})")
            print("="*40)
        
        X_scaled = X.copy()
        
        for col, params in self.params_.items():
            if col not in X.columns:
                if self.verbose:
                    print(f"  ⚠️ Column '{col}' not found, skipping")
                continue
            
            if self.method == 'standard':
                # Apply (x - mean) / std
                mean = params['mean']
                std = params['std']
                X_scaled[col] = (X[col] - mean) / std
            
            elif self.method == 'minmax':
                # Apply (x - min) / (max - min)
                min_val = params['min']
                max_val = params['max']
                range_val = max_val - min_val
                if range_val == 0:
                    X_scaled[col] = 0
                else:
                    X_scaled[col] = (X[col] - min_val) / range_val
            
            elif self.method == 'robust':
                # Apply (x - median) / IQR
                median = params['median']
                iqr = params['iqr']
                if iqr == 0:
                    X_scaled[col] = 0
                else:
                    X_scaled[col] = (X[col] - median) / iqr
            
            # Downcast to float32 to save memory
            X_scaled[col] = X_scaled[col].astype(np.float32)
            
            if self.verbose:
                print(f"  → Scaled '{col}'")
        
        if self.verbose:
            print("="*40)
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Training data
        
        Returns:
            Scaled training data
        """
        self.fit(X)
        return self.transform(X)