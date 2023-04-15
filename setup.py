from setuptools import setup, find_packages

setup(
    name="random_forest_experiment",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "jupyter==1.0.0",
        "pandas==1.5.3",
        "numpy==1.24.2",
        "scikit-learn==1.2.2",
        "shap==0.41.0",
        "mlflow==2.2.2",
        "category_encoders==2.6.0"
    ],
    entry_points={
        "console_scripts": [
            "random_forest_experiment=random_forest_experiment.main:main"
        ]
    }
)
