#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages, Extension

from Cython.Build import cythonize


def build_extensions():
    extensions = [
        Extension(
            'scify._machine',
            ['scify/_machine.pyx'],
        ),
        Extension(
            'scify.specials.debye',
            ['scify/specials/debye.pyx'],
        )
    ]

    directives = {'language_level': '3', 'linetrace': True}
    return cythonize(extensions, compiler_directives=directives, language='c')


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'cython >=0.29',
    'numpy >=1.15',
]

test_requirements = ['pytest', 'pytest-cov']

setup(
    author="Daniel Bok",
    author_email='daniel.bok@outlook.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Scientific functions for Python",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='scify',
    name='scify',
    packages=find_packages(include=['scify']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/DanielBok/scify',
    version='0.1.0',
    zip_safe=False,
    ext_modules=build_extensions(),
)
