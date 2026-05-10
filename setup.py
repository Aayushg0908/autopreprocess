from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="autopreprocess-lite",
    version="0.1.1",
    author="Ayush Gupta",
    author_email="guptaaayush0908@gmail.com",
    description="Automatic Data Preprocessing Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "autopreprocess": ["*.py", "core/*.py", "detectors/*.py", "utils/*.py"],
    },
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
    ],
)