from setuptools import setup, find_packages

setup(
    name='master-thesis',
    version='0.0.1',
    packages=find_packages(include=[
        'master_thesis',
        'master_thesis.*'
    ]),
    install_requires=[],
)
