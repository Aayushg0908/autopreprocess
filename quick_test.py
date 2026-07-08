import pandas as pd
from autopreprocess import AutoClean

# Create sample data
df = pd.DataFrame({
    'age': [25, 30, 45, 22, 35, 28, 40, 33, 29, 31],
    'income': [50000, 60000, 75000, 48000, 68000, 52000, 100000, 55000, 62000, 58000],
    'city': ['NYC', 'LA', 'CHI', 'NYC', 'LA', 'NYC', 'CHI', 'LA', 'NYC', 'CHI'],
    'price': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
})

print("Testing AutoPreprocess...")

pipeline = AutoClean(
    data=df,
    target='price',
    test_size=0.2,
    verbose=True
)

X_train, X_test, y_train, y_test = pipeline.preprocess()

print("\n✅ Test passed!")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")