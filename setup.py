import pathlib
from setuptools import setup, find_packages
# Get the long description from the README file
long_description = (pathlib.Path(__file__).parent.resolve() / "README.md").read_text(encoding="utf-8")

# Prepare the packages and requirements
packages = find_packages(where="src")
requires = [
    "numpy",
    "pandas",
    "scipy",
    "librosa",
    "matplotlib",
    "scikit-learn",
    "sympy",
    #add all pips at the end of the 

]

# Prepare the setup
setup(
    name="VuVoPy",
    version="0.1.0",
    description="Voice features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RichardLadislav/VuVoPy",
    author="Richard Ladislav",
    author_email="230106@vut.cz",
    packages=packages,
    package_data={"": ["LICENSE"]},
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=requires,
    python_requires=">=3.7",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Operating System :: OS Independent"
    ]
)