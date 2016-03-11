from __future__ import division
import commonfate
import numpy
import pytest
import itertools
import operator


@pytest.fixture(params=[0.5, 1])
def length(request):
    return request.param * 44100


@pytest.fixture
def signal(channels, length):
    return numpy.squeeze(numpy.random.random((length, channels)))


@pytest.fixture(params=[1, 2])
def framelength(request):
    return request.param * 512


@pytest.fixture(params=[2, 4])
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
    X = commonfate.transform.forward(x, framelength, hopsize)
    y = commonfate.transform.inverse(
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

    # transform to spectrogram
    X = commonfate.transform.forward(x, framelength, hopsize)
    Z = commonfate.transform.forward(X, W, mhop, real=False)

    # first compute STFT
    Y = commonfate.transform.inverse(
        Z, fdim=2, hop=mhop, shape=X.shape, real=False
    )
    # then back to time domain
    y = commonfate.transform.inverse(
        Y, fdim=1, hop=hopsize, shape=x.shape
    )

    error = numpy.sqrt(numpy.mean((x - y) ** 2))
    print error

    assert error < 1e-8
