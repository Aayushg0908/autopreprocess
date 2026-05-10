import pandas as pd
from utils.memory import reduce_memory_usage

# Create sample data
df = pd.DataFrame({
    'age': [25, 30, 45, 22, 35],
    'salary': [50000, 60000, 75000, 48000, 68000],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
})

print("Original dtypes:")
print(df.dtypes)
print("\n" + "="*40 + "\n")

# Optimize memory
df_optimized = reduce_memory_usage(df, verbose=True)

print("\nOptimized dtypes:")
print(df_optimized.dtypes)