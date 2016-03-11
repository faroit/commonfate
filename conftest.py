import pytest


@pytest.fixture(params=[True, False])
def opt_einsum(request, monkeypatch):
    if not request.param:
        monkeypatch.delattr("opt_einsum.contract")


@pytest.fixture(params=[1, 2])
def channels(request):
    return request.param
