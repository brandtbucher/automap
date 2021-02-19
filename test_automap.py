import pickle
import typing

import hypothesis
import pytest

import automap


Keys = typing.Set[typing.Hashable]


@hypothesis.given(keys=hypothesis.infer)
def test_auto_map___len__(keys: Keys) -> None:
    assert len(automap.AutoMap(keys)) == len(keys)


@hypothesis.given(keys=hypothesis.infer, others=hypothesis.infer)
def test_auto_map___contains__(keys: Keys, others: Keys) -> None:
    a = automap.AutoMap(keys)
    for key in keys:
        assert key in a
    others -= keys
    for key in others:
        assert key not in a


@hypothesis.given(keys=hypothesis.infer, others=hypothesis.infer)
def test_auto_map___getitem__(keys: Keys, others: Keys) -> None:
    a = automap.AutoMap(keys)
    for index, key in enumerate(keys):
        assert a[key] == index
    others -= keys
    for key in others:
        with pytest.raises(KeyError):
            a[key]


@hypothesis.given(keys=hypothesis.infer)
def test_auto_map___hash__(keys: Keys) -> None:
    assert hash(automap.FrozenAutoMap(keys)) == hash(automap.FrozenAutoMap(keys))


@hypothesis.given(keys=hypothesis.infer)
def test_auto_map___iter__(keys: Keys) -> None:
    assert [*automap.AutoMap(keys)] == [*keys]


@hypothesis.given(keys=hypothesis.infer)
def test_auto_map___reversed__(keys: Keys) -> None:
    assert [*reversed(automap.AutoMap(keys))] == [*reversed([*keys])]


@hypothesis.given(keys=hypothesis.infer)
def test_auto_map_add(keys: Keys) -> None:
    a = automap.AutoMap()
    for l, key in enumerate(keys):
        assert a.add(key) is None
        assert len(a) == l + 1
        assert a[key] == l


@hypothesis.given(keys=hypothesis.infer)
def test_pickle(keys: Keys) -> None:
    try:
        hypothesis.assume(pickle.loads(pickle.dumps(keys)) == keys)
    except (TypeError, pickle.PicklingError):
        hypothesis.assume(False)
    a = automap.AutoMap(keys)
    assert pickle.loads(pickle.dumps(a)) == a


@hypothesis.given(keys=hypothesis.infer)
def test_issue_3(keys: Keys) -> None:
    hypothesis.assume(keys)
    key = keys.pop()
    a = automap.AutoMap(keys)
    a |= (key,)
    with pytest.raises(ValueError):
        a |= (key,)
