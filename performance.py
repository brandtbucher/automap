from functools import reduce
from multiprocessing import Pool
from statistics import stdev, harmonic_mean
from random import random
from sys import argv, getsizeof
from timeit import Timer
from itertools import product
from operator import mul

from automap import FrozenAutoMap


namespace = {"FrozenAutoMap": FrozenAutoMap}

create_a = Timer("FrozenAutoMap(keys)", globals=namespace)
create_d = Timer("{k: i for i, k in enumerate(keys)}", globals=namespace)
access_a = Timer("for key in a: a[key]", "a = FrozenAutoMap(keys)", globals=namespace)
access_d = Timer(
    "for key in d: d[key]", "d = {k: i for i, k in enumerate(keys)}", globals=namespace
)


def do_work(info):

    kind, power, factor = info

    items = factor * 10 ** power
    namespace["keys"] = [*{kind(random()) for _ in range(items)}]
    iterations = max(create_a.autorange()[0], create_d.autorange()[0])
    create = create_a.timeit(iterations) / create_d.timeit(iterations)
    size = getsizeof(FrozenAutoMap(namespace["keys"])) / getsizeof(
        {k: i for i, k in enumerate(namespace["keys"])}
    )
    iterations = max(access_a.autorange()[0], access_d.autorange()[0])
    access = access_a.timeit(iterations) / access_d.timeit(iterations)

    return items, create, access, size


print("TYPE\tITEMS\tCREATE\tACCESS\tSIZE")


def geometric_mean(xs):
    return reduce(mul, xs) ** (1 / len(xs))


with Pool() as pool:
    for kind in (str,):
        total_create = []
        total_access = []
        total_size = []
        for items, create, access, size in pool.imap(
            do_work, product((kind,), range(6), range(1, 10))
        ):
            print(
                f"{kind.__name__}\t{items:,}\t{create-1:+.0%}\t{access-1:+.0%}\t{size-1:+.0%}",
                flush=True,
            )
            total_create.append(create)
            total_access.append(access)
            total_size.append(size)
        print(
            f"{kind.__name__}\tMEAN\t{geometric_mean(total_create)-1:+.0%}\t{geometric_mean(total_access)-1:+.0%}\t{geometric_mean(total_size)-1:+.0%}",
            flush=True,
        )
