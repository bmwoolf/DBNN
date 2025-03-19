from setuptools import setup, find_packages

setup(
    name="dbnn",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "matplotlib>=3.8.0",
    ],
)
