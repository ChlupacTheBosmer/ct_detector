from setuptools import setup, find_packages

setup(
    name="ct_detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics",
        "numpy",
        "Pillow",
        "pandas",
        "openpyxl",
        "tabulate"
    ]
)