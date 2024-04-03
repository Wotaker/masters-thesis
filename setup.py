from setuptools import setup, find_packages

setup(
    name='master-thesis',
    version='0.0.1',
    packages=find_packages(include=[
        'master_thesis',
        'master_thesis.*'
    ]),
    install_requires=[
        'karateclub==1.3.3',
        'matplotlib',
        'numpy',
        'networkx==3.1',
        'networkit==11.0',
        'pandas',
        'pyyaml',
        'scipy==1.11.1',
        'scikit-learn',
        'seaborn',
        'torch==2.0.1',
        'torch-geometric==2.4.0',
        'torch-scatter==2.1.2',
        'tqdm',
        'tensorboard'
    ],
)
