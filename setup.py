from setuptools import setup, find_packages

setup(
    name='ezml',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn'
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
        'lightgbm'
    ],
    description='ezml is a lightweight, easy-to-use AutoML library that lets anyone train machine learning models in just a few lines of code — no deep ML knowledge required.',
    author='Ajay',
    python_requires='>=3.8',
)
