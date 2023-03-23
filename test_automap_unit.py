import typing
import pickle
import pytest
import numpy as np

from automap import AutoMap
from automap import FrozenAutoMap
from automap import NonUniqueError


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


def test_fam_contains():
    x = []
    fam = FrozenAutoMap(("a", "b", "c"))
    assert (x in fam.values()) == False
    # NOTE: exercise x to force seg fault
    assert len(x) == 0


# ------------------------------------------------------------------------------


def test_fam_constructor_array_int_a1():
    a1 = np.array((10, 20, 30), dtype=np.int64)
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)


def test_fam_constructor_array_int_a2():
    a1 = np.array((10, 20, 30), dtype=np.int32)
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)


def test_fam_constructor_array_int_b():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64).reshape(2, 2)
    a1.flags.writeable = False
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)


def test_fam_constructor_array_int_c():
    a1 = np.array((10, 20, 30), dtype=np.int8)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    for k in a1:
        assert k in fam


def test_fam_constructor_array_int_d():
    a1 = np.array((-2, -1, 1, 2), dtype=np.int8)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    for k in a1:
        assert k in fam


# def test_fam_constructor_array_a3():
#     a1 = np.array(("a", "bb", "ccc"))
#     with pytest.raises(TypeError):
#         fam = FrozenAutoMap(a1)

# ------------------------------------------------------------------------------


def test_fam_constructor_array_float_a():
    a1 = np.array((1.2, 8.8, 1.2))
    a1.flags.writeable = False
    with pytest.raises(NonUniqueError):
        fam = FrozenAutoMap(a1)


# ------------------------------------------------------------------------------


def test_fam_constructor_array_dt64_a():
    a1 = np.array(("2022-01", "2023-05"), dtype=np.datetime64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert fam[np.datetime64("2023-05")] == 1
    # assert np.datetime64('2022-05') in a1


# ------------------------------------------------------------------------------


def test_fam_constructor_array_unicode_a():
    a1 = np.array(("a", "b", "a"))
    a1.flags.writeable = False
    with pytest.raises(NonUniqueError):
        fam = FrozenAutoMap(a1)


def test_fam_constructor_array_unicode_b():
    a1 = np.array(("a", "bb", "ccc"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    for k in a1:
        assert k in fam


def test_fam_copy_array_unicode_a():
    a1 = np.array(("a", "ccc", "bb"))
    a1.flags.writeable = False
    fam1 = FrozenAutoMap(a1)
    fam2 = FrozenAutoMap(fam1)
    assert fam2["a"] == 0
    assert fam2["ccc"] == 1
    assert fam2["bb"] == 2


# ------------------------------------------------------------------------------


def test_fam_constructor_array_bytes_a():
    a1 = np.array((b"a", b"b", b"c"))
    with pytest.raises(TypeError):
        fam = FrozenAutoMap(a1)


def test_fam_constructor_array_bytes_b():
    a1 = np.array((b"aaa", b"b", b"aaa"))
    a1.flags.writeable = False
    with pytest.raises(NonUniqueError):
        fam = FrozenAutoMap(a1)


def test_fam_constructor_array_bytes_c():
    a1 = np.array((b"aaa", b"b", b"cc"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert fam[b"aaa"] == 0
    assert fam[b"b"] == 1
    assert fam[b"cc"] == 2


def test_fam_copy_array_bytes_a():
    a1 = np.array((b"a", b"ccc", b"bb"))
    a1.flags.writeable = False
    fam1 = FrozenAutoMap(a1)
    fam2 = FrozenAutoMap(fam1)
    assert fam2[b"a"] == 0
    assert fam2[b"ccc"] == 1
    assert fam2[b"bb"] == 2


# ------------------------------------------------------------------------------


def test_fam_array_len_a():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert len(fam) == 4


def test_fam_array_len_b():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert fam[10] == 0
    assert fam[20] == 1
    assert fam[30] == 2
    assert fam[40] == 3


# ------------------------------------------------------------------------------


def test_fam_array_int_get_a():
    a1 = np.array((1, 100, 300, 4000), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 0


def test_fam_array_int_get_b():
    a1 = np.array((1, 100, 300, 4000), dtype=np.int32)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 0
    assert fam.get(1.1) is None


def test_fam_array_int_get_c1():
    a1 = np.array((1, 5, 10, 20), dtype=np.int16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(20.0) == 3


def test_fam_array_int_get_c2():
    a1 = np.array((1,), dtype=np.int16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    for k in a1:
        assert k in fam


def test_fam_array_int_get_c3():
    a1 = np.array((19037,), dtype=np.int16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    for k in a1:
        assert k in fam


def test_fam_array_int_get_d():
    a1 = np.array((1, 5, 10, 20), dtype=np.int8)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(20.0) == 3
    assert fam.get(20.1) is None


def test_fam_array_int_get_e1():
    a1 = np.array([2147483647], dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(2147483647) == 0
    assert fam.get(a1[0]) == 0


# NOTE: this fails on Ubuntu, Windows presently
# def test_fam_array_int_get_e2():
#     a1 = np.array([2147483648], dtype=np.int64)
#     a1.flags.writeable = False
#     fam = FrozenAutoMap(a1)

#     assert fam.get("f") is None
#     assert fam.get(2147483648) == 0
#     assert fam.get(a1[0]) == 0


def test_fam_array_int_get_f():
    ctype = np.int64
    a1 = np.array([np.iinfo(ctype).min, np.iinfo(ctype).max], dtype=ctype)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert list(fam.values()) == []

    assert fam.get("f") is None
    assert fam.get(a1[0]) == 0
    assert fam.get(a1[1]) == 1



# ------------------------------------------------------------------------------


def test_fam_array_uint_get_a():
    a1 = np.array((1, 100, 300, 4000), dtype=np.uint64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 0
    assert fam.get(True) == 0
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 0

    for k in a1:
        assert k in fam


def test_fam_array_uint_get_b():
    a1 = np.arange(0, 100, dtype=np.uint32)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 1
    assert fam.get(True) == 1
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 1

    for k in a1:
        assert k in fam


def test_fam_array_uint_get_c():
    a1 = np.arange(0, 100, dtype=np.uint16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 1
    assert fam.get(True) == 1
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 1

    for k in a1:
        assert k in fam


def test_fam_array_uint_get_d():
    a1 = np.arange(0, 100, dtype=np.uint8)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1) == 1
    assert fam.get(True) == 1
    assert fam.get(a1[2]) == 2
    assert fam.get(1.0) == 1

    for k in a1:
        assert k in fam


def test_fam_array_uint_get_e():
    a1 = np.array((1,), dtype=np.uint16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    for k in a1:
        assert k in fam


# ------------------------------------------------------------------------------


def test_fam_array_float_get_a():
    a1 = np.array((1.5, 10.2, 8.8), dtype=np.float64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(1.5) == 0
    assert fam.get(10.2) == 1
    assert fam.get(a1[1]) == 1
    assert fam.get(8.8) == 2


def test_fam_array_float_get_b():
    a1 = np.array((1.5, 10.2, 8.8), dtype=np.float32)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    # assert fam.get(1.5) == 0
    assert fam.get(a1[0]) == 0
    assert fam.get(a1[1]) == 1
    assert fam.get(a1[2]) == 2


def test_fam_array_float_get_c():
    a1 = np.array((1.5, 10.2, 8.8), dtype=np.float16)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("f") is None
    assert fam.get(a1[0]) == 0
    assert fam.get(a1[1]) == 1
    assert fam.get(a1[2]) == 2


# ------------------------------------------------------------------------------


def test_fam_array_unicode_get_a():
    a1 = np.array(("bb", "a", "ccc"))
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)

    assert fam.get("a") == 1
    assert fam.get("bb") == 0
    assert fam.get("ccc") == 2
    assert fam.get(None) is None
    assert fam.get(3.2) is None
    assert fam.get("cc") is None
    assert fam.get("cccc") is None


# ------------------------------------------------------------------------------


def test_fam_array_values_a():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.values()) == [0, 1, 2, 3]


def test_fam_array_keys_a():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    fam = FrozenAutoMap(a1)
    assert list(fam.keys()) == [10, 20, 30, 40]


def test_fam_array_items_a():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
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


# ------------------------------------------------------------------------------


def test_am_array_constructor_a():
    a1 = np.array(("a", "b", "c"))
    a1.flags.writeable = False
    am1 = AutoMap(a1)


def test_am_array_constructor_b():
    a1 = np.array(("2022-01", "2023-05"), dtype=np.datetime64)
    a1.flags.writeable = False
    am1 = AutoMap(a1)
    assert am1[np.datetime64("2023-05")] == 1


def test_am_array_constructor_c():
    a1 = np.array((10, 20, 30, 40), dtype=np.int64)
    a1.flags.writeable = False
    am = AutoMap(a1)
    am.update((60, 80))
    am.add(90)
    assert list(am.keys()) == [10, 20, 30, 40, 60, 80, 90]


# ------------------------------------------------------------------------------


def test_fam_array_pickle_a():
    a1 = np.array(("a", "b", "c", "d"))
    a1.flags.writeable = False
    fam1 = FrozenAutoMap(a1)

    fam2 = pickle.loads(pickle.dumps(fam1))

    # import ipdb

    # ipdb.set_trace()
