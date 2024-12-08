from setuptools import setup, find_packages

setup(
    name="engine",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "click>=8.0.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
    ],
    entry_points={
        'console_scripts': [
            'engine=engine.cli.main:main',
        ],
    },
)