from setuptools import setup, find_packages

setup(
    name='bindgar',
    version='0.1.0',
    packages=find_packages(where='src'),
    install_requires=[
        'numpy',
        'matplotlib',
        'rebound',
        'pyyaml',
        # Add other dependencies as needed
    ],
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'bindgar=bindgar.cli:main',
        ],
    },
)