import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

n = 10000

# -------------------------
# BASIC COLUMNS
# -------------------------
customer_id = np.arange(10000, 10000+n)

# Age (right-skewed)
age = np.random.gamma(shape=2, scale=10, size=n) + 18
age = np.clip(age, 18, 90)

# Add outliers
age[:5] = np.random.randint(100, 120, 5)

# Income (right-skewed)
income = np.random.lognormal(mean=10, sigma=0.7, size=n)
income = np.clip(income, 20000, 500000)

# Add extreme outliers
income[:10] = np.random.randint(1_000_000, 2_000_000, 10)

# Education (1–5)
education = np.random.choice([1,2,3,4,5], n, p=[0.2,0.25,0.3,0.15,0.1])

# Rating (uniform)
rating = np.random.choice([1,2,3,4,5], n)

# Cities (50)
cities = [f"City_{i}" for i in range(50)]
city = np.random.choice(cities, n)

# States
states = ["CA","NY","TX","FL","WA","IL","AZ","NV","OH","GA"]
state = np.random.choice(states, n)

# Binary
is_active = np.random.choice(["Yes","No"], n, p=[0.7,0.3])

# Dates
start_date = datetime(2020,1,1)
signup_date = [start_date + timedelta(days=np.random.randint(0,2000)) for _ in range(n)]

last_purchase_date = []
for i in range(n):
    if is_active[i] == "No":
        last_purchase_date.append(None)
    else:
        last_purchase_date.append(signup_date[i] + timedelta(days=np.random.randint(1,1000)))

# Purchase frequency (normal)
purchase_frequency = np.random.normal(20, 10, n)
purchase_frequency = np.clip(purchase_frequency, 0, 100)

# Correlated total_spent
total_spent = purchase_frequency * np.random.uniform(100, 300, n)

# Add outliers
total_spent[:15] = np.random.randint(100000, 200000, 15)

# High-cardinality categories
categories = [f"Category_{i}" for i in range(200)]
product_category = np.random.choice(categories, n)

# Constant
constant_col = ["TEST"] * n

# Date recorded (constant-ish)
date_recorded = [datetime(2025,1,1)] * n

# -------------------------
# CREATE DATAFRAME
# -------------------------
df = pd.DataFrame({
    "customer_id": customer_id,
    "age": age,
    "income": income,
    "education_level": education,
    "rating": rating,
    "city": city,
    "state": state,
    "is_active": is_active,
    "signup_date": signup_date,
    "last_purchase_date": last_purchase_date,
    "total_spent": total_spent,
    "purchase_frequency": purchase_frequency,
    "product_category": product_category,
    "constant_col": constant_col,
    "date_recorded": date_recorded
})

# -------------------------
# MISSING VALUES
# -------------------------

# MCAR
df.loc[np.random.choice(n, int(0.02*n)), "age"] = np.nan
df.loc[np.random.choice(n, int(0.01*n)), "rating"] = np.nan

# MAR (income depends on education)
mask = (df["education_level"] <= 2)
df.loc[mask & (np.random.rand(n) < 0.1), "income"] = np.nan

# MNAR (inactive users)
df.loc[df["is_active"]=="No", "last_purchase_date"] = np.nan

# Other missing
df.loc[np.random.choice(n, int(0.15*n)), "city"] = np.nan
df.loc[np.random.choice(n, int(0.05*n)), "state"] = np.nan
df.loc[np.random.choice(n, int(0.02*n)), "purchase_frequency"] = np.nan
df.loc[np.random.choice(n, int(0.20*n)), "product_category"] = np.nan

# -------------------------
# DUPLICATES
# -------------------------
duplicates = df.sample(50)
df = pd.concat([df, duplicates], ignore_index=True)

# -------------------------
# SAVE
# -------------------------
df.to_csv("test_dataset.csv", index=False)

print("Dataset created:", df.shape)