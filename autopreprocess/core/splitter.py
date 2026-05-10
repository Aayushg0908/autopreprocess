"""
Splitter - Split data into train/test sets with multiple strategies

Key principle: SPLIT FIRST before any preprocessing to prevent data leakage
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from typing import Tuple, Union, Optional

class DataSplitter:
    """
    Split data into training and testing sets.
    
    Usage:
        splitter = DataSplitter(test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = splitter.split(X, y)
    
    Strategies:
        - 'random': Random shuffle (default for regression)
        - 'stratified': Preserve class distribution (for classification)
        - 'time_series': No shuffle, chronological order
    """
    
    def __init__(self, 
                 test_size: float = 0.2,
                 train_size: Optional[float] = None,
                 random_state: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize splitter.
        
        Args:
            test_size: Proportion for testing (0.2 = 20% test, 80% train)
            train_size: Optional proportion for training (overrides test_size)
            random_state: Seed for reproducibility
            verbose: Print split info
        """
        # Handle train_size priority
        if train_size is not None:
            self.test_size = 1 - train_size
            self.train_size = train_size
        else:
            self.test_size = test_size
            self.train_size = 1 - test_size
        
        self.random_state = random_state
        self.verbose = verbose
    def _drop_nan_target(self, X, y):
        """Remove rows where target is NaN"""
        if isinstance(y, pd.Series):
            mask = y.notna()
            if self.verbose and (~mask).sum() > 0:
                print(f"   🗑️ Dropped { (~mask).sum()} rows with NaN target")
            
            if isinstance(X, pd.DataFrame):
                X = X[mask]
            else:
                X = X[mask]
            y = y[mask]
        
        return X, y
    def split(self, 
              X: Union[pd.DataFrame, np.ndarray],
              y: Union[pd.Series, np.ndarray],
              strategy: str = 'random') -> Tuple:
        """
        Split data into train and test sets.
        
        Args:
            X: Features
            y: Target
            strategy: 'random', 'stratified', or 'time_series'
        
        Returns:
            (X_train, X_test, y_train, y_test)
        """
           # Drop NaN targets
        X, y = self._drop_nan_target(X, y)
        
        # Input validation
        if len(X) != len(y):
            raise ValueError(f"X length ({len(X)}) != y length ({len(y)})")
        
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        
        if self.verbose:
            print(f"\n✂️  SPLITTING DATA")
            print("="*40)
            print(f"   Strategy: {strategy}")
            print(f"   Train size: {self.train_size:.1%} ({int(len(X) * self.train_size)} rows)")
            print(f"   Test size: {self.test_size:.1%} ({int(len(X) * self.test_size)} rows)")
        
        # Choose strategy
        if strategy == 'stratified':
            X_train, X_test, y_train, y_test = self._stratified_split(X, y)
        elif strategy == 'time_series':
            X_train, X_test, y_train, y_test = self._time_series_split(X, y)
        else:  # 'random'
            X_train, X_test, y_train, y_test = self._random_split(X, y)
        
        # Print class distribution for stratified
        if strategy == 'stratified' and self.verbose:
            self._print_class_distribution(y_train, y_test)
        
        if self.verbose:
            print("="*40)
        
        return X_train, X_test, y_train, y_test
    
    def _random_split(self, X, y) -> Tuple:
        """Random shuffle split using sklearn"""
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
    
    def _stratified_split(self, X, y) -> Tuple:
        """
        Stratified split - preserves class distribution.
        Only works for classification (discrete y values).
        """
        # Check if y is continuous (regression problem)
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 20:
            print("   ⚠️ Warning: Stratified split on continuous target. Using random split instead.")
            return self._random_split(X, y)
        
        # Use sklearn's StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        train_idx, test_idx = next(sss.split(X, y))
        
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
        else:
            X_train = X[train_idx]
            X_test = X[test_idx]
        
        if isinstance(y, pd.Series):
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
        else:
            y_train = y[train_idx]
            y_test = y[test_idx]
        
        return X_train, X_test, y_train, y_test
    
    def _time_series_split(self, X, y) -> Tuple:
        """
        Time series split - no shuffle, chronological order.
        Assumes data is already sorted by date.
        """
        split_idx = int(len(X) * self.train_size)
        
        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
        else:
            X_train = X[:split_idx]
            X_test = X[split_idx:]
        
        if isinstance(y, pd.Series):
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
        else:
            y_train = y[:split_idx]
            y_test = y[split_idx:]
        
        if self.verbose:
            print(f"   📅 Time series: first {split_idx} rows for train")
        
        return X_train, X_test, y_train, y_test
    
    def _print_class_distribution(self, y_train, y_test):
        """Print class distribution for stratified split"""
        print("\n   📊 Class distribution:")
        
        train_counts = y_train.value_counts(normalize=True)
        test_counts = y_test.value_counts(normalize=True)
        
        for cls in train_counts.index:
            print(f"      Class {cls}: train={train_counts[cls]:.2%}, test={test_counts.get(cls, 0):.2%}")
    
    def split_dataframe(self, 
                        df: pd.DataFrame,
                        target_col: str,
                        strategy: str = 'random') -> Tuple:
        """
        Convenience method: split DataFrame into X, y first, then split.
        
        Args:
            df: Full DataFrame
            target_col: Name of target column
            strategy: 'random', 'stratified', or 'time_series'
        
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return self.split(X, y, strategy)