from datetime import date, datetime
from timeit import Timer, timeit
from typing import Hashable, Iterable, Set, Type, TypeVar, FrozenSet, Tuple

from hypothesis import assume, given, infer, seed, settings
from hypothesis.strategies import from_type, frozensets, integers, sampled_from
from pytest import mark, raises

from automap import AutoMap


Atom = TypeVar("Atom", bytes, complex, date, datetime, float, int, str)
CompositeAtom = TypeVar("CompositeAtom", Atom, Tuple[Atom, ...], FrozenSet[Atom])

keys = frozensets(from_type(CompositeAtom))


@given(keys=keys)
def test_auto_map___len__(keys: FrozenSet[Hashable]):
    assert len(AutoMap(keys)) == len(keys)


@given(keys=keys, others=keys)
def test_auto_map___contains__(keys: FrozenSet[Hashable], others: FrozenSet[Hashable]):
    a = AutoMap(keys)
    for key in keys:
        assert key in a
    others -= keys
    for key in others:
        assert key not in a


@given(keys=keys, others=keys)
def test_auto_map___getitem__(keys: FrozenSet[Hashable], others: FrozenSet[Hashable]):
    a = AutoMap(keys)
    for index, key in enumerate(keys):
        assert a[key] == index
    others -= keys
    for key in others:
        with raises(KeyError):
            a[key]


# @given(keys=keys)
# def test_auto_map___hash__(keys: FrozenSet[Hashable]):
#     assert hash(AutoMap(keys)) == hash(AutoMap(keys))


@given(keys=keys)
def test_auto_map___iter__(keys: FrozenSet[Hashable]):
    assert [*AutoMap(keys)] == [*keys]


@given(keys=keys, key=from_type(CompositeAtom))
def test_issue_3(keys: FrozenSet[Hashable], key: Hashable):
    assume(key not in keys)
    a = AutoMap(keys)
    a |= (key,)
    with raises(ValueError):
        a |= (key,)
