from setuptools import find_packages, setup


setup(
    name='gt-env',
    packages=find_packages(),
    install_requires=[
        'nashpy',
        'matplotlib',
        'notebook',
        'ipywidgets',
        'bigtree',
        'graphviz'
    ],
)