import os
import sys

from setuptools import setup, find_packages, Extension

ENV = 'DEV'
argv = sys.argv
for e in argv:
    if e.startswith('--env'):
        _, ENV = e.upper().split('=')
        argv.remove(e)

IS_DEV_MODE = ENV == 'DEV'

try:
    from Cython.Build import cythonize
    from Cython.Compiler import Options

    USE_CYTHON = True
    Options.annotate = IS_DEV_MODE

except ImportError:
    def cythonize(ext, *args, **kwargs):
        return ext


    USE_CYTHON = False


    class Options:
        pass


def build_extensions():
    extensions = []
    for root, _, files in os.walk("scify"):
        path_parts = os.path.normcase(root).split(os.sep)
        for file in files:
            fn, ext = os.path.splitext(file)

            if ext == '.pyx':
                module_path = '.'.join([*path_parts, fn])
                pyx_c_file_path = os.path.join(*path_parts, fn + ('.pyx' if USE_CYTHON else '.c'))
                extensions.append(Extension(module_path,
                                            [pyx_c_file_path],
                                            language='c'))

    directives = {'language_level': '3', 'linetrace': IS_DEV_MODE}
    return cythonize(extensions, compiler_directives=directives)


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
    packages=find_packages(include=['scify', 'scify.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/DanielBok/scify',
    version='0.1.0',
    zip_safe=False,
    ext_modules=build_extensions(),
)
