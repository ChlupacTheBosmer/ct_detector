from setuptools import setup

setup(
    name="ct_detector",
    version="0.1.0",
    packages=["ct_detector"],
    install_requires=[
        "ultralytics",
        "numpy",
        "Pillow",
        "pandas",
        "openpyxl",
        "lap",
        "tabulate"
    ]
)