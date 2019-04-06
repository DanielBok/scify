======================================
Scify: Scientific functions for Python
======================================

Scify: The package
~~~~~~~~~~~~~~~~~~

`scify` is a collection of scientific functions. All of the functions here can also be found in the `GSL <https://www.gnu.org/software/gsl/>`_ package or `Boost <https://www.boost.org/>`_.

`scify` is for people who want access to these functions but do not want to go through the trouble of installing a C/C++ compiler (especially on Windows) and compiling those libraries so that they can be used in Python.

The goal of the package is for the functions to be available across platforms with minimal installation frustrations.

Why this package?
~~~~~~~~~~~~~~~~~

Admittedly for most people, `scipy <https://www.scipy.org/>`_ would be able to satisfy most of their scientific computing need. If you find that `scipy` already does the job, there's no need to look into `scify`.

Use this only if you want to try the functions inside and you don't want to go through the hassle of installing compilers and stuff into your PC.

TODO
~~~~

There's a tons of stuff to do for this package.

Namely:

- Add all the functions
    - [ ] Airy
    - [ ] Bessel
    - [ ] Beta
    - [x] Clausen
    - [ ] Coupling
    - [ ] Cubature
    - [ ] Dawson
    - [x] Debye
    - [x] Dilog
    - [ ] Ellint
    - [ ] Error
    - [ ] Expint
    - [ ] Fermi Dirac
    - [ ] Gamma
    - [ ] Gegenbauer
    - [ ] Hyperg
    - [ ] Laguerre
    - [ ] Lambert
    - [ ] Legendre
    - [ ] Log
    - [ ] Misc
    - [ ] Poly
    - [ ] Psi
    - [ ] Qrng
    - [ ] Rng
    - [ ] Synchrotron
    - [ ] Transport
    - [ ] Trig
    - [ ] Zeta
- [ ] Write the docs
- [ ] Travis test
- Build
    - [ ] Build the wheels for all platform
    - [ ] Conda build

