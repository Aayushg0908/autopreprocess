"""
Selector - Feature selection module to remove irrelevant/redundant features

Methods:
- variance: Remove constant/low variance columns
- correlation: Remove one of highly correlated pairs
- importance: Keep top K features by RandomForest importance
- mutual_info: Keep top K features by mutual information score
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class FeatureSelector:
    """
    Automatic feature selection for ML models.
    
    Usage:
        selector = FeatureSelector(method='variance', verbose=True)
        X_selected = selector.fit_transform(X, y)
        
        # Get info about dropped features
        print(selector.dropped_features_)
        print(selector.selected_features_)
    """
    
    def __init__(self,
                 method: str = 'variance',
                 variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.9,
                 top_k: Optional[int] = None,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize FeatureSelector.
        
        Args:
            method: 'variance', 'correlation', 'importance', or 'mutual_info'
            variance_threshold: Remove features with variance below this
            correlation_threshold: Remove one feature from pairs above this
            top_k: Number of top features to keep (if None, keep all passing filters)
            random_state: Seed for reproducibility
            verbose: Print progress
        """
        self.method = method
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.top_k = top_k
        self.random_state = random_state
        self.verbose = verbose
        
        # Storage
        self.selected_features_ = []
        self.dropped_features_ = []
        self.feature_importances_ = {}
        self.correlated_pairs_ = []
    
    def _detect_problem_type(self, y: pd.Series) -> str:
        """Detect classification vs regression"""
        if y.nunique() <= 20 and y.dtype in ['int64', 'int32', 'object', 'category']:
            return 'classification'
        return 'regression'
    
    def _variance_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove constant or low variance columns"""
        if self.verbose:
            print("\n  📊 Variance Threshold Selection")
            print(f"     Threshold: {self.variance_threshold}")
        
        # Calculate variance for each column
        variances = X.var()
        low_variance_cols = variances[variances < self.variance_threshold].index.tolist()
        
        # Also check for constant columns (variance = 0)
        constant_cols = X.columns[X.nunique() == 1].tolist()
        
        all_dropped = list(set(low_variance_cols + constant_cols))
        
        if all_dropped:
            self.dropped_features_.extend(all_dropped)
            X_selected = X.drop(columns=all_dropped)
            if self.verbose:
                print(f"     Dropped {len(all_dropped)} columns (low variance/constant)")
        else:
            X_selected = X.copy()
            if self.verbose:
                print(f"     No low variance columns found")
        
        return X_selected
    
    def _correlation_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove one feature from highly correlated pairs"""
        if self.verbose:
            print(f"\n  📊 Correlation Removal")
            print(f"     Threshold: {self.correlation_threshold}")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Upper triangle mask
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find correlated pairs
        correlated_pairs = []
        to_drop = set()
        
        for col in upper_tri.columns:
            high_corr = upper_tri[col][upper_tri[col] > self.correlation_threshold]
            for idx in high_corr.index:
                correlated_pairs.append((col, idx, high_corr[idx]))
                # Keep the one with higher variance (more information)
                if X[col].var() < X[idx].var():
                    to_drop.add(col)
                else:
                    to_drop.add(idx)
        
        self.correlated_pairs_ = correlated_pairs
        
        if to_drop:
            self.dropped_features_.extend(list(to_drop))
            X_selected = X.drop(columns=list(to_drop))
            if self.verbose:
                print(f"     Found {len(correlated_pairs)} correlated pairs")
                print(f"     Dropped {len(to_drop)} columns")
        else:
            X_selected = X.copy()
            if self.verbose:
                print(f"     No high correlations found")
        
        return X_selected
    
    def _importance_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Keep top K features by RandomForest importance"""
        if self.verbose:
            print(f"\n  📊 Feature Importance Selection")
            print(f"     Top K: {self.top_k if self.top_k else 'auto'}")
        
        problem_type = self._detect_problem_type(y)
        
        # Select model based on problem type
        if problem_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        
        # Fit and get importances
        model.fit(X, y)
        importances = model.feature_importances_
        
        # Store feature importances
        self.feature_importances_ = dict(zip(X.columns, importances))
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)
        
        # Determine number to keep
        if self.top_k:
            keep_count = min(self.top_k, len(X.columns))
        else:
            # Keep features with importance > mean importance
            mean_importance = np.mean(importances)
            keep_count = sum(1 for imp in importances if imp > mean_importance)
            keep_count = max(1, keep_count)  # Keep at least 1
        
        selected = [f[0] for f in sorted_features[:keep_count]]
        dropped = [f[0] for f in sorted_features[keep_count:]]
        
        self.selected_features_ = selected
        self.dropped_features_.extend(dropped)
        
        if self.verbose:
            print(f"     Kept {len(selected)} features")
            print(f"     Dropped {len(dropped)} features")
            print(f"     Top 3: {selected[:3]}")
        
        return X[selected]
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Keep top K features by mutual information score"""
        if self.verbose:
            print(f"\n  📊 Mutual Information Selection")
            print(f"     Top K: {self.top_k if self.top_k else 'auto'}")
        
        problem_type = self._detect_problem_type(y)
        
        # Calculate mutual information
        if problem_type == 'classification':
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        
        # Store scores
        self.feature_importances_ = dict(zip(X.columns, mi_scores))
        
        # Sort by score
        sorted_features = sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)
        
        # Determine number to keep
        if self.top_k:
            keep_count = min(self.top_k, len(X.columns))
        else:
            # Keep features with positive mutual information
            keep_count = sum(1 for score in mi_scores if score > 0)
            keep_count = max(1, keep_count)
        
        selected = [f[0] for f in sorted_features[:keep_count]]
        dropped = [f[0] for f in sorted_features[keep_count:]]
        
        self.selected_features_ = selected
        self.dropped_features_.extend(dropped)
        
        if self.verbose:
            print(f"     Kept {len(selected)} features")
            print(f"     Dropped {len(dropped)} features")
        
        return X[selected]
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit selector and determine which features to keep.
        
        Args:
            X: Features
            y: Target
        
        Returns:
            self
        """
        if self.verbose:
            print(f"\n🔧 FITTING FEATURE SELECTOR")
            print("="*40)
            print(f"   Method: {self.method}")
            print(f"   Input features: {X.shape[1]}")
        
        # Start with all features
        X_current = X.copy()
        
        # Step 1: Always apply variance threshold (safe first step)
        X_current = self._variance_selection(X_current)
        
        # Step 2: Apply correlation removal if method supports
        if self.method == 'correlation':
            X_current = self._correlation_selection(X_current)
        
        # Step 3: Apply main selection method
        if self.method == 'importance':
            X_final = self._importance_selection(X_current, y)
        elif self.method == 'mutual_info':
            X_final = self._mutual_info_selection(X_current, y)
        elif self.method == 'correlation':
            # Already applied correlation, now keep all remaining
            X_final = X_current
            self.selected_features_ = list(X_current.columns)
        else:  # 'variance'
            X_final = X_current
            self.selected_features_ = list(X_current.columns)
        
        # Remove duplicates from dropped list
        self.dropped_features_ = list(set(self.dropped_features_))
        
        if self.verbose:
            print("="*40)
            print(f"✅ Selection complete")
            print(f"   Kept: {len(self.selected_features_)}")
            print(f"   Dropped: {len(self.dropped_features_)}")
            print("="*40)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using selected features.
        
        Args:
            X: Features to transform
        
        Returns:
            DataFrame with only selected features
        """
        if not self.selected_features_:
            raise RuntimeError("Selector not fitted. Call fit() first.")
        
        # Keep only selected features that exist
        keep_cols = [col for col in self.selected_features_ if col in X.columns]
        
        if len(keep_cols) != len(self.selected_features_):
            missing = set(self.selected_features_) - set(keep_cols)
            if self.verbose:
                print(f"   ⚠️ Missing {len(missing)} features in transform data")
        
        return X[keep_cols]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get selection summary"""
        return {
            'method': self.method,
            'total_features': len(self.selected_features_) + len(self.dropped_features_),
            'kept_features': len(self.selected_features_),
            'dropped_features': len(self.dropped_features_),
            'selected': self.selected_features_[:10],  # Top 10
            'dropped': self.dropped_features_[:10],
            'correlated_pairs': len(self.correlated_pairs_)
        }
    
    def print_summary(self):
        """Print readable summary"""
        summary = self.get_summary()
        print("\n" + "="*40)
        print("📊 FEATURE SELECTION SUMMARY")
        print("="*40)
        print(f"   Method: {summary['method']}")
        print(f"   Total features: {summary['total_features']}")
        print(f"   Kept: {summary['kept_features']}")
        print(f"   Dropped: {summary['dropped_features']}")
        print(f"   Correlated pairs removed: {summary['correlated_pairs']}")
        print("="*40)