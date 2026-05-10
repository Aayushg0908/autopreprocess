"""
Pipeline - Main AutoClean class that orchestrates all preprocessing steps

Usage:
    from autopreprocess import AutoClean
    
    cleaner = AutoClean('data.csv', target='price')
    X_train, X_test, y_train, y_test = cleaner.preprocess()
    
    # For new data
    X_new_clean = cleaner.predict_ready_data(X_new)
"""

import pandas as pd
import numpy as np
import time
from typing import Optional, Union, Tuple, Dict, Any, List

from .splitter import DataSplitter
from .cleaner import DataCleaner
from .encoder import Encoder
from .scaler import Scaler
from .selector import FeatureSelector
from ..utils.memory import reduce_memory_usage
from ..utils.cache import save_pipeline, load_pipeline


class AutoClean:
    """
    Automatic Data Preprocessing Library
    
    Main class that orchestrates:
    - Train/test split
    - Data cleaning (missing, outliers, duplicates)
    - Categorical encoding
    - Feature scaling
    """
    
    def __init__(self,
                data: Union[str, pd.DataFrame],
                target: str,
                test_size: float = 0.2,
                random_state: Optional[int] = 42,
                scaling: bool = True,
                scaling_method: str = 'standard',
                encoding: str = 'auto',
                outlier_method: str = 'iqr',
                handle_missing: bool = True,
                handle_outliers: bool = True,
                remove_duplicates: bool = True,
                drop_useless: bool = True,
                ignore_cols: Optional[List[str]] = None,
                verbose: bool = True,
                # NEW: Feature selection parameters
                feature_selection: bool = False,
                selection_method: str = 'variance',
                top_k: Optional[int] = None):
        """
        Initialize AutoClean pipeline.
        
        Args:
            data: CSV file path or pandas DataFrame
            target: Target column name
            test_size: Proportion for testing (default 0.2)
            random_state: Seed for reproducibility
            scaling: Whether to scale numeric features
            scaling_method: 'standard', 'minmax', or 'robust'
            encoding: 'auto' (auto-detect) or 'onehot'/'frequency'
            outlier_method: 'iqr' for now
            handle_missing: Handle missing values
            handle_outliers: Handle outliers
            remove_duplicates: Remove duplicate rows
            drop_useless: Drop useless columns (IDs, constants)
            ignore_cols: List of columns to ignore (don't preprocess)
            verbose: Print progress
        """
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.scaling = scaling
        self.scaling_method = scaling_method
        self.encoding = encoding
        self.outlier_method = outlier_method
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.remove_duplicates = remove_duplicates
        self.drop_useless = drop_useless
        self.ignore_cols = ignore_cols or []
        self.verbose = verbose
        self.feature_selection = feature_selection
        self.selection_method = selection_method
        self.top_k = top_k
        
        # Storage for fitted components
        self.splitter_ = None
        self.cleaner_ = None
        self.encoder_ = None
        self.scaler_ = None
        
        # Column tracking
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        self.datetime_cols_ = []
        self.dropped_cols_ = []
        self.encoded_cols_ = []
        self.scaled_cols_ = []
        
        # Problem type
        self.problem_type_ = None  # 'classification' or 'regression'
        
        # Timing
        self.timing_ = {}
        
        # Load data
        self.df_ = self._load_data(data)
        
        if self.verbose:
            print("\n" + "="*60)
            print("🤖 AUTOCLEAN PIPELINE INITIALIZED")
            print("="*60)
            print(f"   Data shape: {self.df_.shape}")
            print(f"   Target: {self.target}")
            print(f"   Test size: {self.test_size}")
    
    def _load_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Load data from CSV or DataFrame"""
        if isinstance(data, str):
            if self.verbose:
                print(f"📂 Loading data from: {data}")
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            if self.verbose:
                print(f"📂 Using provided DataFrame")
            df = data.copy()
        else:
            raise ValueError("data must be CSV file path or pandas DataFrame")
        
        # Check target exists
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in data")
        
        # Remove ignore columns
        if self.ignore_cols:
            df = df.drop(columns=[col for col in self.ignore_cols if col in df.columns])
            if self.verbose:
                print(f"   Ignored columns: {self.ignore_cols}")
        
        # Memory optimization
        df = reduce_memory_usage(df, verbose=False)
        
        return df
    
    def _detect_problem_type(self, y: pd.Series) -> str:
        """Detect if problem is classification or regression"""
        unique_count = y.nunique()
        
        # If few unique values and integer-like → classification
        if unique_count <= 20 and y.dtype in ['int64', 'int32', 'object', 'category']:
            return 'classification'
        else:
            return 'regression'
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'AutoClean':
        """
        Fit preprocessing pipeline on training data.
        
        Args:
            X: Training features
            y: Training target
        
        Returns:
            self (fitted pipeline)
        """
        if self.verbose:
            print("\n🔧 FITTING PIPELINE")
            print("="*40)
        
        # Detect problem type
        self.problem_type_ = self._detect_problem_type(y)
        if self.verbose:
            print(f"   Problem type: {self.problem_type_}")
        
        # Initialize components
        self.cleaner_ = DataCleaner()
        self.encoder_ = Encoder(verbose=self.verbose)
        self.scaler_ = Scaler(method=self.scaling_method, verbose=self.verbose)
        
        # Step 1: Clean data
        if self.verbose:
            print("\n📋 Step 1: Cleaning")
        clean_result = self.cleaner_.clean_dataframe(
            pd.concat([X, y.to_frame()], axis=1),
            handle_missing=self.handle_missing,
            handle_outliers=self.handle_outliers,
            remove_duplicates=self.remove_duplicates,
            drop_useless=self.drop_useless,
            verbose=False
        )
        X_clean = clean_result['cleaned_data'].drop(columns=[self.target])
        
        # Track dropped columns
        if 'useless_columns_dropped' in clean_result:
            self.dropped_cols_.extend(clean_result['useless_columns_dropped'])
        
        # Step 2: Encode categoricals
        if self.verbose:
            print("\n📋 Step 2: Encoding")
        self.encoder_.fit(X_clean)
        X_encoded = self.encoder_.transform(X_clean)
        
        # Track encoded columns
        self.categorical_cols_ = self.encoder_.categorical_cols or []
        self.encoded_cols_ = list(X_encoded.columns)
        
        # Step 3: Scale numeric features
        if self.scaling:
            if self.verbose:
                print("\n📋 Step 3: Scaling")
            self.scaler_.fit(X_encoded)
            X_scaled = self.scaler_.transform(X_encoded)
            self.scaled_cols_ = self.scaler_.numeric_cols or []
        else:
            X_scaled = X_encoded
      # NEW: Feature selection
        if self.feature_selection:
            if self.verbose:
                print("\n📋 Step 4: Feature Selection")
            
            self.selector_ = FeatureSelector(
                method=self.selection_method,
                top_k=self.top_k,
                verbose=self.verbose
            )
            X_scaled = self.selector_.fit_transform(X_scaled, y)
            self.selected_features_ = self.selector_.selected_features_
        else:
            self.selected_features_ = list(X_scaled.columns)      
                
        # Store feature names
        self.feature_names_ = list(X_scaled.columns)
        self.feature_names_ = self.selected_features_
        if self.verbose:
            print("="*40)
            print("✅ Fitting complete")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted pipeline.
        
        Args:
            X: Data to transform (test or new data)
        
        Returns:
            Transformed DataFrame
        """
        if self.cleaner_ is None:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        
        if self.verbose:
            print("\n🔄 TRANSFORMING DATA")
            print("="*40)
        
        # Clean
        X_clean = X.copy()
        
        # Drop columns that were dropped during fit
        for col in self.dropped_cols_:
            if col in X_clean.columns:
                X_clean = X_clean.drop(columns=[col])
        
        # Encode
        X_encoded = self.encoder_.transform(X_clean)
        
        # Scale
        if self.scaling:
            X_scaled = self.scaler_.transform(X_encoded)
        else:
            X_scaled = X_encoded
        
        # Ensure columns match training (add missing, drop extra)
        for col in self.feature_names_:
            if col not in X_scaled.columns:
                X_scaled[col] = 0
        
        X_scaled = X_scaled[self.feature_names_]
        if self.feature_selection and hasattr(self, 'selector_'):
            X_scaled = self.selector_.transform(X_scaled)
        for col in self.feature_names_:
          if col not in X_scaled.columns:
              X_scaled[col] = 0
        X_scaled = X_scaled[self.feature_names_]
      
        if self.verbose:
            print(f"   Output shape: {X_scaled.shape}")
            print("="*40)
        
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)
    
    def preprocess(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Main method: Split, fit on train, transform both train and test.
        
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*60)
            print("🚀 STARTING PREPROCESSING PIPELINE")
            print("="*60)
        
        # Step 1: Split data FIRST (prevent leakage)
        if self.verbose:
            print("\n📋 Step 0: Train/Test Split")
        
        self.splitter_ = DataSplitter(
            test_size=self.test_size,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        X = self.df_.drop(columns=[self.target])
        y = self.df_[self.target]
        
        # Use stratified split for classification
        strategy = 'stratified' if self.problem_type_ == 'classification' else 'random'
        
        X_train, X_test, y_train, y_test = self.splitter_.split(X, y, strategy=strategy)
        
        if self.verbose:
            print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Step 2: Fit on training data
        self.fit(X_train, y_train)
        
        # Step 3: Transform both train and test
        X_train_processed = self.transform(X_train)
        X_test_processed = self.transform(X_test)
        
        self.timing_['total'] = time.time() - start_time
        
        if self.verbose:
            print("\n" + "="*60)
            print("✅ PREPROCESSING COMPLETE")
            print("="*60)
            print(f"   Total time: {self.timing_['total']:.2f} seconds")
            print(f"   Train shape: {X_train_processed.shape}")
            print(f"   Test shape: {X_test_processed.shape}")
            print("="*60)
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def predict_ready_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform completely new data for inference/deployment.
        
        Args:
            new_data: New data (same columns as training)
        
        Returns:
            Preprocessed data ready for model prediction
        """
        if self.verbose:
            print("\n🔮 PREPARING DATA FOR PREDICTION")
            print("="*40)
        
        return self.transform(new_data)
    
    def summary(self) -> Dict[str, Any]:
        """Return pipeline summary"""
        return {
            'target': self.target,
            'problem_type': self.problem_type_,
            'original_shape': self.df_.shape,
            'feature_count': len(self.feature_names_) if hasattr(self, 'feature_names_') else 0,
            'numeric_cols': self.scaled_cols_,
            'categorical_cols': self.categorical_cols_,
            'dropped_cols': self.dropped_cols_,
            'timing': self.timing_,
            'feature_selection': self.feature_selection,
            'selection_method': self.selection_method if self.feature_selection else None           
        }
    
    def print_summary(self):
        """Print readable summary"""
        print("\n" + "="*60)
        print("📊 PIPELINE SUMMARY")
        print("="*60)
        print(f"   Target: {self.target}")
        print(f"   Problem type: {self.problem_type_}")
        print(f"   Features: {len(self.feature_names_)}")
        print(f"   Numeric columns scaled: {len(self.scaled_cols_)}")
        print(f"   Categorical columns encoded: {len(self.categorical_cols_)}")
        print(f"   Columns dropped: {len(self.dropped_cols_)}")
        if self.feature_selection:
          print(f"   Feature selection: {self.selection_method} (kept {len(self.feature_names_)})")        
        print(f"   Total time: {self.timing_.get('total', 0):.2f}s")
        print("="*60)
    
    def save(self, filepath: str):
        """Save pipeline to disk"""
        save_pipeline(self, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'AutoClean':
        """Load pipeline from disk"""
        return load_pipeline(filepath)