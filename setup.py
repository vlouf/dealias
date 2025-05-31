from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def parse_requirements(filename):
    with open(filename, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


REQUIRED = parse_requirements("requirements.txt")
setup(
    name="unravel",  # Required
    version="1.4.0",  # Required
    author="Valentin Louf",  # Optional
    author_email="valentin.louf@bom.gov.au",  # Optional
    description="A dealiasing technique for Doppler radar velocity.",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/vlouf/dealias",  # Optional
    project_urls={
        "Documentation": "https://github.com/vlouf/dealias#readme",
        "Source": "https://github.com/vlouf/dealias",
        "Tracker": "https://github.com/vlouf/dealias/issues",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    keywords="radar weather meteorology dealiasing Doppler",  # Optional
    packages=find_packages(exclude=["contrib", "docs", "tests"]),  # Required
    install_requires=REQUIRED,
    license="Apache-2.0",
    python_requires=">=3.8",
)
