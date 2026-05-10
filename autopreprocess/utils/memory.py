"""
Memory Optimization Utility
Reduces RAM usage by downcasting numeric columns to smaller data types
"""

import pandas as pd
import numpy as np


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of a pandas DataFrame by downcasting numeric columns

    Args:
        df: Input DataFrame
        verbose: Print memory savings report

    Returns:
        Optimized DataFrame
    """

    # Initial memory usage (in MB)
    start_memory = df.memory_usage(deep=True).sum() / 1024**2

    # Copy dataframe
    df_optimized = df.copy()

    for col in df_optimized.columns:
        col_data = df_optimized[col]

        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(col_data):
            continue

        col_type = col_data.dtype

        # Handle integers
        if pd.api.types.is_integer_dtype(col_type):
            col_min = col_data.min()
            col_max = col_data.max()

            # Unsigned integers (only positive values)
            if col_min >= 0:
                if col_max <= np.iinfo(np.uint8).max:
                    df_optimized[col] = col_data.astype(np.uint8)
                elif col_max <= np.iinfo(np.uint16).max:
                    df_optimized[col] = col_data.astype(np.uint16)
                elif col_max <= np.iinfo(np.uint32).max:
                    df_optimized[col] = col_data.astype(np.uint32)
                else:
                    df_optimized[col] = col_data.astype(np.uint64)

            # Signed integers (negative values possible)
            else:
                if np.iinfo(np.int8).min <= col_min <= col_max <= np.iinfo(np.int8).max:
                    df_optimized[col] = col_data.astype(np.int8)
                elif np.iinfo(np.int16).min <= col_min <= col_max <= np.iinfo(np.int16).max:
                    df_optimized[col] = col_data.astype(np.int16)
                elif np.iinfo(np.int32).min <= col_min <= col_max <= np.iinfo(np.int32).max:
                    df_optimized[col] = col_data.astype(np.int32)
                else:
                    df_optimized[col] = col_data.astype(np.int64)

        # Handle floats
        elif pd.api.types.is_float_dtype(col_type):
            # Downcast float64 → float32
            if col_type == "float64":
                df_optimized[col] = col_data.astype(np.float32)

    # Final memory usage
    end_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
    saved_memory = start_memory - end_memory
    reduction_pct = (saved_memory / start_memory) * 100 if start_memory > 0 else 0

    if verbose:
        print(f"💾 Memory: {start_memory:.2f} MB → {end_memory:.2f} MB")
        print(f"💾 Saved: {saved_memory:.2f} MB ({reduction_pct:.1f}% reduction)")

    return df_optimized