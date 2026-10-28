import pytest

from mars.tubes.protocol import SPLITS, TubesSpec, check_split, physical_seed


def test_protocol_splits_and_versioning():
    for name, ids in SPLITS.items():
        assert all(set(ids).isdisjoint(other) for key, other in SPLITS.items() if key != name)
    check_split("train", 0, 0)
    with pytest.raises(ValueError):
        check_split("test", 30000, 0, heldout=True)
    with pytest.raises(ValueError):
        check_split("train", 0, 1)
    assert physical_seed(42, 0) != physical_seed(43, 0)
    assert TubesSpec().fingerprint != TubesSpec(clearance=.003).fingerprint
    assert TubesSpec().stages == ("grasp", "insert") * 3 + ("home",)
    with pytest.raises(ValueError):
        TubesSpec(clearance=.01)
