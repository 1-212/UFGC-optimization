from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('gc_optimization_app/requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='gc_optimization',
    version='0.1.0',
    description='GC Optimization Project for Gas Chromatography Data Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='GC Optimization Team',
    author_email='',
    url='',
    packages=find_packages(),
    package_data={
        'gc_optimization_app': ['requirements.txt'],
    },
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    python_requires='>=3.9',
    entry_points={
        'console_scripts': [
            'gc-optimization=gc_optimization_app.main:main',
        ],
    },
)