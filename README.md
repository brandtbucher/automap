<div align=justify>

<div align=center>

automap
=======

[![latest version](https://img.shields.io/github/release-pre/brandtbucher/automap.svg?style=for-the-badge&label=latest)![latest release date](https://img.shields.io/github/release-date-pre/brandtbucher/automap.svg?style=for-the-badge&label=released)](https://github.com/brandtbucher/automap/releases)[![build status](https://img.shields.io/travis/com/brandtbucher/automap/master.svg?style=for-the-badge)](https://travis-ci.com/brandtbucher/automap/branches)[![issues](https://img.shields.io/github/issues-raw/brandtbucher/automap.svg?label=issues&style=for-the-badge)](https://github.com/brandtbucher/automap/issues)

<br>

</div>

`automap` is a Python package containing high-performance autoincremented integer-valued mappings.

To install, just run `pip install automap`.

Examples
--------

`automap` objects are sort of like "inverse sequences". They come in two variants:

### FrozenAutoMap

```py
>>> from automap import FrozenAutoMap
```

`FrozenAutoMap` objects are immutable. They can be constructed from any iterable of hashable, unique keys.


```py
>>> a = FrozenAutoMap("AAA")
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'A'
>>> a = FrozenAutoMap("ABC")
>>> a
automap.FrozenAutoMap(['A', 'B', 'C'])
```

The values are integers, incrementing according to the order of the original keys:

```py
>>> a["A"]
0
>>> a["C"]
2
>>> a["X"]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
KeyError: 'X'
```

The full `Mapping` interface is provided:

```py
>>> a.keys()
('A', 'B', 'C')
>>> a.values()
[0, 1, 2]
>>> a.items()
(('A', 0), ('B', 1), ('C', 2))
>>> a.get("X", 42)
42
>>> "B" in a
True
>>> list(a)
['A', 'B', 'C']
```

They may also be combined with each other using the `|` operator:

```py
>>> b = FrozenAutoMap(range(5))
>>> c = FrozenAutoMap(range(5, 10))
>>> b | c
automap.FrozenAutoMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> b |= c  # Note that b is reassigned, not mutated!
>>> b
automap.FrozenAutoMap([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

### AutoMap

```py
>>> from automap import AutoMap
```

Unlike `FrozenAutoMap` objects, `AutoMap` objects can grow; new keys may be
added, but existing ones may not be deleted or changed.

```py
>>> d = AutoMap("ABC")
>>> d
automap.AutoMap(['A', 'B', 'C'])
>>> d |= "DEF"  # Here, d *is* mutated!
automap.AutoMap(['A', 'B', 'C', 'D', 'E', 'F'])
```

Performance
-----------

Preliminary tests show random string-keyed `AutoMap` objects being created
60-80% faster and accessed 10-30% faster than the equivalent `dict`
construction. They tend to take up ~50% more RAM on average. You can run `python
performance.py` from this repo to see the comparison on your machine.

More details on the design can be found in `automap.c`.

</div>
