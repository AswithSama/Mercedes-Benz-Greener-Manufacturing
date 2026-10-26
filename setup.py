from setuptools import find_packages, setup

setup(
    name="mercedes-benz-greener-manufacturing",
    version="0.1.0",
    description="Reproducible manufacturing test-time regression research pipeline",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24,<3",
        "pandas>=2.0,<3",
        "scikit-learn>=1.3,<2",
        "flask>=3.0,<4",
    ],
)
