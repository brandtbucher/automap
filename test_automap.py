from datetime import date, datetime
from timeit import Timer, timeit
from typing import Hashable, Iterable, Set, Type, TypeVar, FrozenSet, Tuple, Union

from hypothesis import assume, given, infer, seed, settings
from hypothesis.strategies import from_type, frozensets, integers, sampled_from
from pytest import mark, raises

from automap import AutoMap, FrozenAutoMap


# Atom should just be Hashable, but hypothesis chokes on NaNs sometimes, so we
# need to exclude float and complex... :(
Atom = Union[bool, int, None, str, bytes, date, datetime]
CompositeAtom = Union[Atom, Tuple[Atom, ...], FrozenSet[Atom]]


@given(keys=infer)
def test_auto_map___len__(keys: FrozenSet[CompositeAtom]):
    assert len(AutoMap(keys)) == len(keys)


@given(keys=infer, others=infer)
def test_auto_map___contains__(
    keys: FrozenSet[CompositeAtom], others: FrozenSet[CompositeAtom]
):
    a = AutoMap(keys)
    for key in keys:
        assert key in a
    others -= keys
    for key in others:
        assert key not in a


@given(keys=infer, others=infer)
def test_auto_map___getitem__(
    keys: FrozenSet[CompositeAtom], others: FrozenSet[CompositeAtom]
):
    a = AutoMap(keys)
    for index, key in enumerate(keys):
        assert a[key] == index
    others -= keys
    for key in others:
        with raises(KeyError):
            a[key]


@given(keys=infer)
def test_auto_map___hash__(keys: FrozenSet[CompositeAtom]):
    assert hash(FrozenAutoMap(keys)) == hash(FrozenAutoMap(keys))


@given(keys=infer)
def test_auto_map___iter__(keys: FrozenSet[CompositeAtom]):
    assert [*AutoMap(keys)] == [*keys]


@given(keys=infer)
def test_auto_map_add(keys: FrozenSet[CompositeAtom]):
    a = AutoMap()
    for l, key in enumerate(keys):
        assert a.add(key) is None
        assert len(a) == l + 1
        assert a[key] == l


@given(keys=infer, key=infer)
def test_issue_3(keys: FrozenSet[CompositeAtom], key: CompositeAtom):
    assume(key not in keys)
    a = AutoMap(keys)
    a |= (key,)
    with raises(ValueError):
        a |= (key,)
