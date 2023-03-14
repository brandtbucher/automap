import typing
import pytest
import numpy as np

from automap import AutoMap
from automap import FrozenAutoMap

# from automap import NonUniqueError


def test_am_extend():
    am1 = AutoMap(("a", "b"))
    am2 = am1 | AutoMap(("c", "d"))
    assert list(am2.keys()) == ["a", "b", "c", "d"]


def test_am_add():
    a = AutoMap()
    for l, key in enumerate(["a", "b", "c", "d"]):
        assert a.add(key) is None
        assert len(a) == l + 1
        assert a[key] == l


def test_contains():
    x = []
    fam = FrozenAutoMap(("a", "b", "c"))
    assert (x in fam.values()) == False
    # NOTE: exercise x to force seg fault
    assert len(x) == 0


def test_constructor_array_a():
    a1 = np.array((10, 20, 30))
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)


def test_constructor_array_b():
    a1 = np.array(("2022-01", "2023-05"), dtype=np.datetime64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert fam[np.datetime64("2023-05")] == 1
    # assert np.datetime64('2022-05') in a1


def test_constructor_array_c():
    a1 = np.array((10, 20, 30, 40)).reshape(2, 2)
    a1.flags.writeable = False
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)


def test_fam_array_len_a():
    a1 = np.array((10, 20, 30, 40))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert len(fam) == 4


def test_fam_array_len_b():
    a1 = np.array((10, 20, 30, 40))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert fam[10] == 0
    assert fam[20] == 1
    assert fam[30] == 2
    assert fam[40] == 3


def test_am_array_raises():
    a1 = np.array((10, 20, 30, 40))
    a1.flags.writeable = False
    am = AutoMap(a1)
    with pytest.raises(NotImplementedError):
        am.update((60, 80))

    with pytest.raises(NotImplementedError):
        am.add(80)


def test_fam_array_values_a():
    a1 = np.array((10, 20, 30, 40))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.values()) == [0, 1, 2, 3]


def test_fam_array_keys_a():
    a1 = np.array((10, 20, 30, 40))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.keys()) == [10, 20, 30, 40]


def test_fam_array_items_a():
    a1 = np.array((10, 20, 30, 40))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.items()) == [(10, 0), (20, 1), (30, 2), (40, 3)]


def test_fam_array_values_b():
    a1 = np.array(("a", "b", "c", "d"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.values()) == [0, 1, 2, 3]


def test_fam_array_keys_b():
    a1 = np.array(("a", "b", "c", "d"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.keys()) == ["a", "b", "c", "d"]


def test_fam_array_items_b():
    a1 = np.array(("a", "b", "c", "d"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.items()) == [("a", 0), ("b", 1), ("c", 2), ("d", 3)]


def test_fam_array_items_c():
    a1 = np.array(("a", "b", "c"))
    a1.flags.writeable = False
    fam1 = FrozenAutoMap(a1)

    fam2 = FrozenAutoMap(fam1)
    assert list(fam2.items()) == [("a", 0), ("b", 1), ("c", 2)]
    assert list(fam1.items()) == [("a", 0), ("b", 1), ("c", 2)]
