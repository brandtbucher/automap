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
    a1 = np.array((10, 20, 30, 40)).reshape(2, 2)
    a1.flags.writeable = False
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)
