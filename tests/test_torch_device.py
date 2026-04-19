from retrieval.torch_device import resolve_torch_device


def test_resolve_explicit_cpu():
    assert resolve_torch_device("cpu") == "cpu"
    assert resolve_torch_device("  CPU  ") == "cpu"
