# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

with open("README.md","r",encoding="utf-8-sig") as f:
    readme=f.read()
setup(
    name="pythainlu",
    version="0.1.dev1",
    description="Thai Natural Language Understanding library",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Wannaphong",
    author_email="wannaphong@kkumail.com",
    url="https://github.com/wannaphongcom/pythainlu",
    packages=find_packages(),
    test_suite="tests",
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=["nltk"],
    license="Apache Software License 2.0"
)