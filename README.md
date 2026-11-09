[![PyPI version](https://badge.fury.io/py/autopreprocess-lite.svg)](https://pypi.org/project/autopreprocess-lite/)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
# AutoPreprocess

**Automatic Data Preprocessing Library for Machine Learning**

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## ✨ Features

- **Automatic column type detection** (numeric, categorical, datetime, useless)
- **Smart missing value handling** (based on missing percentage)
- **Outlier detection & capping** (IQR method)
- **Intelligent encoding** (One-hot, Frequency, Target encoding)
- **Feature scaling** (Standard, MinMax, Robust)
- **Feature selection** (Variance, Correlation, Importance, Mutual Info)
- **Train/Test split** (Random, Stratified, Time series)
- **Save & load pipeline** for production deployment
- **Zero data leakage** (fit only on training data)

## 🚀 Quick Start

```python
from autopreprocess import AutoClean

# One line to preprocess everything
pipeline = AutoClean('data.csv', target='price')
X_train, X_test, y_train, y_test = pipeline.preprocess()

# For new predictions
X_new_clean = pipeline.predict_ready_data(X_new)

# Save for later
pipeline.save('my_pipeline.pkl')

# Load and use
loaded = AutoClean.load('my_pipeline.pkl')
```

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/autopreprocess)](https://github.com/YOUR_USERNAME/autopreprocess/stargazers)
[![PyPI version](https://badge.fury.io/py/autopreprocess-lite.svg)](https://pypi.org/project/autopreprocess-lite/)

## 🖥️ Web UI

A one-page web app (Flask + Jinja) for running the library without writing code:
upload a CSV, pick the target, tune the cleaning/scaling/selection options, and
download the processed train/test sets and the fitted pipeline.

```bash
pip install -r ui/requirements.txt
python ui/app.py          # then open http://127.0.0.1:5000
```

Set `PORT` to use a different port. Click **Try with sample dataset** to see it work without your own data.
