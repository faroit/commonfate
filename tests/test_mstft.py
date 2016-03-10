from __future__ import division
from pycfm import cft
import numpy
import pytest
import itertools
import operator


@pytest.fixture(params=[1, 2, 4])
def channels(request):
    return request.param


@pytest.fixture(params=[1, 2])
def length(request):
    return request.param * 44100


@pytest.fixture
def signal(channels, length):
    return numpy.squeeze(numpy.random.random((length, channels)))


@pytest.fixture(params=[1, 2, 4])
def framelength(request):
    return request.param * 512


@pytest.fixture(params=[2, 4, 8])
def hopsize(framelength, request):
    return framelength // request.param


@pytest.fixture(params=itertools.combinations((8, 10, 20), 2))
# create many different shapes of dimension 2
def W(request):
    return request.param


@pytest.fixture(params=[2, 4])
def mhop(W, request):
    d = (request.param, request.param)
    return tuple(map(operator.floordiv, W, d))


def test_2d(channels, signal, framelength, hopsize):
    """
    Test if transform-inverse identity holds for a simple spectogram
    """
    x = signal

    # transform to spectogram
    X = cft.stft(x, framelength, hopsize)
    y = cft.istft(
        X, fdim=1, hop=hopsize, shape=x.shape
    )

    error = numpy.sqrt(numpy.mean((x - y) ** 2))
    print error

    assert error < 1e-8


def test_grid(channels, signal, framelength, hopsize, W, mhop):
    """
    Test if transform-inverse identity holds for the tensor case
    """
    x = signal

    # transform to spectogram
    X = cft.stft(x, framelength, hopsize)
    Z = cft.stft(X, W, mhop, real=False)

    # first compute back STFT
    Y = cft.istft(
        Z, fdim=2, hop=mhop, shape=X.shape, real=False
    )
    y = cft.istft(
        Y, fdim=1, hop=hopsize, shape=x.shape
    )

    error = numpy.sqrt(numpy.mean((x - y) ** 2))
    print error

    assert error < 1e-8
