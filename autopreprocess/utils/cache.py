"""
Cache Utility - Save and Load Preprocessing Pipelines

What it does:
1. save_pipeline() - Saves fitted preprocessor to disk
2. load_pipeline() - Loads saved preprocessor from disk

Why needed:
- Preprocessing steps like scaling learn parameters (mean, std)
- These parameters MUST be reused on new data
- Saving ensures consistency between train and prediction
"""

import joblib
import os
from typing import Any

def save_pipeline(pipeline: Any, filepath: str) -> None:
    """
    Save a fitted preprocessing pipeline to disk
    
    Args:
        pipeline: Fitted AutoClean object (after calling .preprocess())
        filepath: Where to save the file (e.g., 'models/preprocessor.pkl')
    
    Example:
        cleaner = AutoClean('train.csv', target='price')
        X_train, X_test, y_train, y_test = cleaner.preprocess()
        save_pipeline(cleaner, 'models/my_preprocessor.pkl')
    
    What gets saved:
        - Fitted scalers (with their mean/std values)
        - Fitted encoders (with their category mappings)
        - Column names and preprocessing steps
    """
    
    # Extract directory from filepath (e.g., 'models/' from 'models/preprocessor.pkl')
    directory = os.path.dirname(filepath)
    
    # Create directory if it doesn't exist
    if directory:  # Only if directory is not empty
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Save using joblib (better for large numpy arrays than pickle)
    joblib.dump(pipeline, filepath)
    print(f"✅ Pipeline saved to: {filepath}")
    
    # Show file size (so user knows it worked)
    file_size = os.path.getsize(filepath) / 1024  # Convert to KB
    if file_size < 1024:
        print(f"📦 File size: {file_size:.2f} KB")
    else:
        print(f"📦 File size: {file_size/1024:.2f} MB")


def load_pipeline(filepath: str) -> Any:
    """
    Load a saved preprocessing pipeline from disk
    
    Args:
        filepath: Path to saved .pkl file
    
    Returns:
        Loaded AutoClean object (already fitted, ready to transform new data)
    
    Example:
        cleaner = load_pipeline('models/my_preprocessor.pkl')
        X_new_clean = cleaner.transform(new_data)  # No refitting!
    
    Important:
        The loaded pipeline is ALREADY fitted.
        Do NOT call .preprocess() again on new data.
        Use .transform(new_data) instead.
    """
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ No pipeline found at: {filepath}")
    
    # Load pipeline
    print(f"📂 Loading pipeline from: {filepath}")
    pipeline = joblib.load(filepath)
    
    # Show file size that was loaded
    file_size = os.path.getsize(filepath) / 1024
    if file_size < 1024:
        print(f"📦 Loaded {file_size:.2f} KB")
    else:
        print(f"📦 Loaded {file_size/1024:.2f} MB")
    
    print("✅ Pipeline loaded successfully!")
    return pipeline


def delete_pipeline(filepath: str) -> None:
    """
    Delete a saved pipeline file (optional utility)
    
    Args:
        filepath: Path to saved .pkl file to delete
    """
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"🗑️ Deleted pipeline: {filepath}")
    else:
        print(f"⚠️ No pipeline found at: {filepath}")