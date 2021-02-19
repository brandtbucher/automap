from datetime import date, datetime
from pickle import dumps, loads
from timeit import Timer
import typing

from hypothesis import assume, given, infer
from pytest import raises

from automap import AutoMap, FrozenAutoMap


Keys = typing.Set[typing.Hashable]


@given(keys=infer)
def test_auto_map___len__(keys: Keys) -> None:
    assert len(AutoMap(keys)) == len(keys)


@given(keys=infer, others=infer)
def test_auto_map___contains__(keys: Keys, others: Keys) -> None:
    a = AutoMap(keys)
    for key in keys:
        assert key in a
    others -= keys
    for key in others:
        assert key not in a


@given(keys=infer, others=infer)
def test_auto_map___getitem__(keys: Keys, others: Keys) -> None:
    a = AutoMap(keys)
    for index, key in enumerate(keys):
        assert a[key] == index
    others -= keys
    for key in others:
        with raises(KeyError):
            a[key]


@given(keys=infer)
def test_auto_map___hash__(keys: Keys) -> None:
    assert hash(FrozenAutoMap(keys)) == hash(FrozenAutoMap(keys))


@given(keys=infer)
def test_auto_map___iter__(keys: Keys) -> None:
    assert [*AutoMap(keys)] == [*keys]


@given(keys=infer)
def test_auto_map___reversed__(keys: Keys) -> None:
    assert [*reversed(AutoMap(keys))] == [*reversed([*keys])]


@given(keys=infer)
def test_auto_map_add(keys: Keys) -> None:
    a = AutoMap()
    for l, key in enumerate(keys):
        assert a.add(key) is None
        assert len(a) == l + 1
        assert a[key] == l


@given(keys=infer)
def test_pickle(keys: Keys) -> None:
    try:
        assume(loads(dumps(keys)) == keys)
    except TypeError:
        assume(False)
    a = AutoMap(keys)
    assert loads(dumps(a)) == a


@given(keys=infer)
def test_issue_3(keys: Keys) -> None:
    assume(keys)
    key = keys.pop()
    a = AutoMap(keys)
    a |= (key,)
    with raises(ValueError):
        a |= (key,)
