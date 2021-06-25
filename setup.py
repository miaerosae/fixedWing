import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fw",
    version="0.1.0",
    url="git@github.com:miaerosae/fixedWing.git",
    author="miae kim",
    description="various control method for fixed-wing UAV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="",
    include_package_data=True,
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "matplotlib"
    ]
)
