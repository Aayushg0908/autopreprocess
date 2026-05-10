import pandas as pd
from utils.cache import save_pipeline, load_pipeline, delete_pipeline

# Create a dummy pipeline (we'll replace with real AutoClean later)
class DummyPipeline:
    def __init__(self):
        self.mean_age = 35
        self.cities = {'NYC': 0, 'LA': 1, 'CHI': 2}
    
    def transform(self, data):
        print("✨ Transforming new data using saved parameters!")
        return data

# Step 1: Create and save pipeline
print("="*40)
print("SAVING PIPELINE")
print("="*40)
my_pipeline = DummyPipeline()
save_pipeline(my_pipeline, 'test_pipeline.pkl')

print("\n" + "="*40)
print("LOADING PIPELINE")
print("="*40)

# Step 2: Load it back
loaded_pipeline = load_pipeline('test_pipeline.pkl')

# Step 3: Verify it works
print(f"\nLoaded pipeline mean_age: {loaded_pipeline.mean_age}")
print(f"Loaded pipeline cities: {loaded_pipeline.cities}")

# Step 4: Test transform
test_data = pd.DataFrame({'name': ['John', 'Jane']})
loaded_pipeline.transform(test_data)

# Step 5: Clean up
print("\n" + "="*40)
print("CLEANING UP")
print("="*40)
delete_pipeline('test_pipeline.pkl')