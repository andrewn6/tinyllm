from setuptools import setup, find_packages

setup(
    name="tinyllm",
    version="0.1.0",
    packages=['tinyllm'] + ['tinyllm.' + pkg for pkg in find_packages(where='tinyllm')],
    package_dir={'': '.'},
    install_requires=[
        'torch',
        'numpy',
        'fastapi',
        'uvicorn',
        'requests',
        'click>=8.0.0'
    ],
    entry_points={
        'console_scripts': [
            'tinyllm=tinyllm.cli.commands:cli',
        ],
    },
)