import numpy as np
import pytest


@pytest.fixture('module')
def data():
    np.random.seed(8)
    return np.random.uniform(-5, 5, 1000000)


@pytest.fixture('module')
def complex_data():
    np.random.seed(8)
    return np.random.uniform(-5, 5, 1000000) + 1j * np.random.uniform(-5, 5, 1000000)
