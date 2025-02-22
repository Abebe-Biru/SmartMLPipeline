from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='smart-ml-pipeline',
    version='0.1.1',
    author='Abebe Biru',
    author_email='smartml.team@gmail.com',
    description='An automated and customizable ML pipeline for classification tasks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/abebe-biru/SmartMLPipeline',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'optuna',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
