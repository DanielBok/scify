# Scify: Scientific functions for Python

## Scify: The package

`scify` is a collection of scientific functions which can be found in packages like [GSL](https://www.gnu.org/software/gsl/) package or [Boost](https://www.boost.org/) but are not provided in `scipy`. 

`scify` is for those who want access to these functions but do not want to go through the trouble of installing a C/C++ compiler (especially on Windows) and compiling those libraries so that they can be used in Python.

The goal of the package is for the functions to be available across platforms with minimal installation frustrations.

## Why this package?

Most commonly used scientific functions are already provided for by [scipy](https://www.scipy.org/). In most cases, it would be better off to use those functions.

Functions here are implemented either because they are not provided for or because they can be broken down to simpler components.

For example, the Dilogarithm function is not provided. Another case is that the Airy function provided by `scipy` returns the first, second and their respective derivatives. However, many times in my usage, I would only need one of them and thus can gain a minor optimization by splitting it up.

In the latter case, it would likely constitute some differences if one is running the functions through a Monte Carlo simulations of sorts. However, before optimizing, I would advise reevaluating the business use case, then profiling the code before working on more involved optimizations like `scify`.  

## TODO

There's a tons of stuff to do for this package.

Namely:

- Add all the functions
    - [x] Airy
    - [ ] Bessel
    - [ ] Beta
    - [x] Clausen
    - [ ] Coupling
    - [ ] Cubature
    - [ ] Dawson
    - [x] Debye
    - [x] Dilog
    - [ ] Gamma
    - [x] Log
    - [ ] Poly
    - [ ] Psi
    - [ ] Qrng
    - [ ] Rng
    - [ ] Trig
    - [ ] Zeta
- [ ] Write the docs
- [ ] Travis test
- Build
    - [ ] Build the wheels for all platform
    - [ ] Conda build

