from __future__ import division
import commonfate as cf
import numpy as np
import pytest
import itertools
import operator


@pytest.fixture(params=[0.5])
def length(rate, request):
    return request.param * rate


@pytest.fixture
def signal(channels, length):
    return np.random.random((length, channels))


@pytest.fixture(params=[2])
def framelength(request):
    return request.param * 512


@pytest.fixture(params=[2])
def hopsize(framelength, request):
    return framelength // request.param


@pytest.fixture(params=itertools.combinations((30, 20, 10), 2))
# create many different shapes of dimension 2
def W(request):
    return request.param


@pytest.fixture(params=[2])
def mhop(W, request):
    d = (request.param, request.param)
    return tuple(map(operator.floordiv, W, d))


@pytest.fixture(params=[16000, 22050])
def rate(request):
    return request.param


def test_reconstruction(
    channels, rate, signal, framelength, hopsize, W, mhop, opt_einsum
):
    """
    Test if transform-inverse identity holds for the tensor case
    """
    components = cf.decompose.process(
        signal,
        rate,
        nb_iter=50,
        nb_components=2,
        n_fft=framelength,
        n_hop=hopsize,
        cft_patch=W,
        cft_hop=mhop,
    )

    # testing shapes
    assert np.sum(components, axis=0).shape == signal.shape

    # testing reconstruction error
    error = np.sqrt(np.mean((np.sum(components, axis=0) - signal) ** 2))
    assert error < 1e-8
