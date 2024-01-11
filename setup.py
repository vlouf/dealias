from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="unravel",  # Required    
    version="1.3.1",  # Required
    description="A dealiasing technique for Doppler radar velocity.",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/vlouf/dealias",  # Optional
    author="Valentin Louf",  # Optional
    author_email="valentin.louf@bom.gov.au",  # Optional
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="radar weather meteorology dealiasing Doppler",  # Optional
    packages=find_packages(exclude=["contrib", "docs", "tests"]),  # Required
    install_requires=["numpy", "numba", "arm_pyart"],  # Optional
    project_urls={  # Optional
        "Bug Reports": "https://github.com/vlouf/dealias/issues",
        "Source": "https://github.com/vlouf/dealias/",
    },
)
