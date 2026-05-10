"""
Encoder - Converts categorical and datetime columns to numeric format

Main features:
- Frequency encoding (replace category with its count)
- One-hot encoding for low-cardinality columns (<10 unique values)
- Unknown value handling (unseen categories → -1)
- Datetime feature extraction (year, month, day, weekday, is_weekend)
- Ordinal encoding with custom mappings

Why this approach:
- Frequency encoding works well for tree-based models (Random Forest, XGBoost)
- One-hot is better for linear models when cardinality is low
- Unknown handling prevents crashes on test data with new categories
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any

class Encoder:
    """
    Encodes categorical and datetime columns for machine learning.
    
    How to use:
        encoder = Encoder(verbose=True)
        X_encoded = encoder.fit_transform(X_train)
        X_test_encoded = encoder.transform(X_test)
    
    What happens during fit():
        - Detects which columns are categorical vs datetime
        - For each categorical column, decides encoding method:
            * <10 unique values → One-hot encoding
            * >=10 unique values → Frequency encoding
        - Stores the frequency mappings for later transform
        - Stores one-hot category lists
    
    What happens during transform():
        - Applies stored mappings to new data
        - Unknown categories (not seen during fit) become -1
        - Datetime columns get expanded into multiple features
    """
    
    def __init__(self, 
                 categorical_cols: Optional[List[str]] = None,
                 ordinal_cols: Optional[List[str]] = None,
                 ordinal_mappings: Optional[Dict[str, List[Any]]] = None,
                 datetime_cols: Optional[List[str]] = None,
                 unknown_value: int = -1,
                 verbose: bool = True):
        """
        Initialize the Encoder.
        
        Parameters:
        -----------
        categorical_cols : list of str, optional
            Names of columns to treat as categorical.
            If None, automatically detects using type_detector.
        
        ordinal_cols : list of str, optional
            Names of columns that have an inherent order.
            Example: education_level (1=HS, 2=College, 3=Grad)
        
        ordinal_mappings : dict, optional
            Custom order for ordinal columns.
            Example: {'education_level': [1, 2, 3, 4, 5]}
        
        datetime_cols : list of str, optional
            Names of columns containing dates/times.
            If None, automatically detects using type_detector.
        
        unknown_value : int, default -1
            Value to use for categories not seen during fit().
            -1 is safe because real frequencies are always >=0.
        
        verbose : bool, default True
            If True, prints progress and decisions.
        """
        self.categorical_cols = categorical_cols
        self.ordinal_cols = ordinal_cols or []
        self.ordinal_mappings = ordinal_mappings or {}
        self.datetime_cols = datetime_cols
        self.unknown_value = unknown_value
        self.verbose = verbose
        
        # Storage for fitted encoders (populated during fit())
        self.freq_maps_ = {}        # {col_name: {category: frequency_count}}
        self.onehot_categories_ = {}  # {col_name: [list_of_categories]}
        self.ordinal_maps_ = {}     # {col_name: {value: encoded_number}}
    
    def _detect_categorical_datetime(self, df: pd.DataFrame) -> None:
        """
        Automatically identify which columns are categorical and which are datetime.
        
        Why: User doesn't have to manually specify column types.
        
        How it works:
            - Uses our previously built TypeDetector class
            - Gets lists of: categorical, discrete_numeric, datetime
            - Discrete numeric (like ratings 1-5) are treated as categorical
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input data to analyze
        
        Returns:
        --------
        None (updates self.categorical_cols and self.datetime_cols)
        """
        from detectors.type_detector import TypeDetector
        detector = TypeDetector()
        analysis = detector.analyze_dataframe(df, verbose=False)
        
        # Combine regular categorical + discrete numeric (e.g., rating 1-5)
        all_categorical = analysis['categorical'] + analysis['discrete_numeric']
        
        # Only set if user didn't provide them
        if self.categorical_cols is None:
            self.categorical_cols = all_categorical
        if self.datetime_cols is None:
            self.datetime_cols = analysis['datetime']
    
    def fit(self, X: pd.DataFrame, y=None) -> 'Encoder':
        """
        Learn encoding mappings from training data.
        
        IMPORTANT: Only call this on training data, NOT on test data.
        Test data should use .transform() with mappings learned here.
        
        What happens:
            1. Detects column types if not specified
            2. For each categorical column, decides encoding method based on cardinality
            3. Stores frequency counts or category lists for later use
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features (the data to learn encodings from)
        
        y : ignored
            Kept for API compatibility (not used in simple encoder)
        
        Returns:
        --------
        self : Encoder
            Returns self (fitted encoder)
        """
        if self.verbose:
            print("\n🔧 FITTING ENCODER")
            print("="*40)
            print(f"   Input shape: {X.shape}")
        
        # Step 1: Detect column types if not provided
        if self.categorical_cols is None or self.datetime_cols is None:
            self._detect_categorical_datetime(X)
            if self.verbose:
                print(f"   Detected categorical: {len(self.categorical_cols)} columns")
                print(f"   Detected datetime: {len(self.datetime_cols)} columns")
        
        # Step 2: Process each categorical column
        for col in self.categorical_cols:
            # Skip if column doesn't exist in the data
            if col not in X.columns:
                if self.verbose:
                    print(f"  ⚠️ Column '{col}' not found in data, skipping")
                continue
            
            # Count how many unique values this column has
            unique_vals = X[col].nunique()
            
            # Decision: One-hot for low cardinality, Frequency for high cardinality
            # Why? One-hot creates many columns (bad for high cardinality)
            # Frequency encoding keeps one column regardless of cardinality
            if unique_vals < 10:
                # ONE-HOT ENCODING
                # Store the list of all possible categories
                categories = X[col].dropna().unique().tolist()
                self.onehot_categories_[col] = categories
                if self.verbose:
                    print(f"  ✅ '{col}': One-hot encoding ({unique_vals} categories)")
            else:
                # FREQUENCY ENCODING
                # Count how many times each category appears
                # Example: {'NYC': 500, 'LA': 300, 'CHI': 200}
                freq_map = X[col].value_counts().to_dict()
                self.freq_maps_[col] = freq_map
                if self.verbose:
                    print(f"  ✅ '{col}': Frequency encoding ({unique_vals} categories)")
        
        # Step 3: Process ordinal columns (user-specified order)
        for col in self.ordinal_cols:
            if col not in X.columns:
                continue
            
            # Check if user provided custom mapping
            if col in self.ordinal_mappings:
                # Example mapping: [1, 2, 3, 4, 5] becomes {1:0, 2:1, 3:2, 4:3, 5:4}
                order = self.ordinal_mappings[col]
                self.ordinal_maps_[col] = {val: idx for idx, val in enumerate(order)}
                if self.verbose:
                    print(f"  ✅ '{col}': Ordinal encoding with custom order")
            else:
                # No mapping provided - will use raw values directly
                self.ordinal_maps_[col] = None
                if self.verbose:
                    print(f"  ✅ '{col}': Ordinal encoding (using raw values)")
        
        # Step 4: Log datetime columns
        if self.datetime_cols and self.verbose:
            print(f"  ✅ Datetime columns to extract: {self.datetime_cols}")
        
        if self.verbose:
            print("="*40)
            print("✅ Fitting complete")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned encodings to transform data.
        
        This is used for BOTH test data and new predictions.
        It uses mappings learned during fit(), so categories not seen
        during training become unknown_value (default -1).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Data to transform (can be test set or new data)
        
        Returns:
        --------
        pd.DataFrame
            Encoded DataFrame with all categorical/datetime columns transformed
            to numeric format.
        """
        if not self.freq_maps_ and not self.onehot_categories_ and not self.ordinal_maps_:
            raise RuntimeError("Encoder not fitted. Call fit() first.")
        
        if self.verbose:
            print("\n🔄 TRANSFORMING DATA")
            print("="*40)
            print(f"   Input shape: {X.shape}")
        
        # List to store all encoded DataFrames (will be concatenated at the end)
        result_dfs = []
        
        # ============================================
        # PROCESS FREQUENCY ENCODED COLUMNS
        # ============================================
        for col, freq_map in self.freq_maps_.items():
            if col not in X.columns:
                if self.verbose:
                    print(f"  ⚠️ Column '{col}' missing, skipping")
                continue
            
            # Map each category to its frequency count
            # Example: 'NYC' → 500, 'LA' → 300
            encoded = X[col].map(freq_map)
            
            # Handle unknown categories (not seen during fit)
            # These become NaN after .map(), so we fill with unknown_value
            encoded = encoded.fillna(self.unknown_value)
            
            # Ensure numeric type and downcast to save memory
            encoded = pd.to_numeric(encoded, errors='coerce').fillna(self.unknown_value)
            encoded = encoded.astype(np.float32)  # float32 uses 4 bytes instead of 8
            
            # Add to results
            result_dfs.append(pd.DataFrame({f"{col}_freq": encoded}, index=X.index))
            
            if self.verbose:
                unknown_count = (encoded == self.unknown_value).sum()
                print(f"  → Frequency encoded '{col}' ({unknown_count} unknown → {self.unknown_value})")
        
        # ============================================
        # PROCESS ONE-HOT ENCODED COLUMNS
        # ============================================
        for col, categories in self.onehot_categories_.items():
            if col not in X.columns:
                continue
            
            # Create a binary column for each category
            # Example: city_LA: 1 if city is LA, else 0
            for cat in categories:
                # Check if this specific category appears in current row
                is_category = (X[col] == cat).astype(np.uint8)
                result_dfs.append(pd.DataFrame({f"{col}_{cat}": is_category}, index=X.index))
            
            if self.verbose:
                print(f"  → One-hot encoded '{col}' → {len(categories)} columns")
        
        # ============================================
        # PROCESS ORDINAL COLUMNS
        # ============================================
        for col, mapping in self.ordinal_maps_.items():
            if col not in X.columns:
                continue
            
            if mapping is not None:
                # Use custom mapping provided by user
                encoded = X[col].map(mapping)
            else:
                # Use raw values (assumed to be already ordinal)
                encoded = pd.to_numeric(X[col], errors='coerce')
            
            # Handle missing or invalid values
            encoded = encoded.fillna(self.unknown_value).astype(np.int8)
            result_dfs.append(pd.DataFrame({f"{col}_ord": encoded}, index=X.index))
            
            if self.verbose:
                print(f"  → Ordinal encoded '{col}'")
        
        # ============================================
        # PROCESS DATETIME COLUMNS
        # ============================================
        for col in self.datetime_cols:
            if col not in X.columns:
                continue
            
            # Convert to datetime (handles various string formats)
            dt = pd.to_datetime(X[col], errors='coerce')
            
            # Extract useful features from dates
            # Why 5 features? These are the most predictive for ML:
            # - year: trend over time
            # - month: seasonal patterns
            # - day: position in month
            # - weekday: day-of-week patterns (Monday vs Sunday)
            # - is_weekend: captures weekend effect
            features = pd.DataFrame({
                f"{col}_year": dt.dt.year,
                f"{col}_month": dt.dt.month,
                f"{col}_day": dt.dt.day,
                f"{col}_weekday": dt.dt.weekday,  # Monday=0, Sunday=6
                f"{col}_is_weekend": (dt.dt.weekday >= 5).astype(np.uint8)
            }, index=X.index)
            
            # Replace missing dates (NaT) with 0
            features = features.fillna(0).astype(np.int16)  # int16 uses 2 bytes
            result_dfs.append(features)
            
            if self.verbose:
                print(f"  → Datetime extracted from '{col}' → 5 features")
        
        # ============================================
        # COMBINE ALL RESULTS
        # ============================================
        if not result_dfs:
            # No categorical/datetime columns, return original
            if self.verbose:
                print("  → No categorical or datetime columns to encode")
            return X.copy()
        
        # Concatenate all encoded DataFrames side-by-side
        result = pd.concat(result_dfs, axis=1)
        
        if self.verbose:
            print(f"  → Final encoded shape: {result.shape}")
            print("="*40)
        
        return result
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit encoder to training data AND transform it in one step.
        
        This is a convenience method that combines fit() and transform().
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        y : ignored
        
        Returns:
        --------
        pd.DataFrame
            Encoded training data
        """
        self.fit(X, y)
        return self.transform(X)