import pytest

from sklearn import datasets

@pytest.fixture
def digits():
    return datasets.load_digits()

# @pytest.fixture

