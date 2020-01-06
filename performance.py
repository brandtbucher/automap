from random import random
from sys import argv, getsizeof
from timeit import Timer

from automap import FrozenAutoMap


namespace = {"FrozenAutoMap": FrozenAutoMap}

create_a = Timer("FrozenAutoMap(keys)", globals=namespace)
create_d = Timer("{k: i for i, k in enumerate(keys)}", globals=namespace)
access_a = Timer("for key in a: a[key]", "a = FrozenAutoMap(keys)", globals=namespace)
access_d = Timer(
    "for key in d: d[key]", "d = {k: i for i, k in enumerate(keys)}", globals=namespace
)

print("ITEMS\tCREATE\tACCESS\tSIZE")

for power in range(0, 6):

    for factor in range(1, 10):

        items = factor * 10 ** power
        namespace["keys"] = [*{str(random()) for _ in range(items)}]
        iterations = max(create_a.autorange()[0], create_d.autorange()[0])
        create = create_a.timeit(iterations) / create_d.timeit(iterations) - 1
        size = (
            getsizeof(FrozenAutoMap(namespace["keys"]))
            / getsizeof({k: i for i, k in enumerate(namespace["keys"])})
            - 1
        )
        iterations = max(access_a.autorange()[0], access_d.autorange()[0])
        access = access_a.timeit(iterations) / access_d.timeit(iterations) - 1
        print(f"{items:,}\t{create:+.0%}\t{access:+.0%}\t{size:+.0%}")
