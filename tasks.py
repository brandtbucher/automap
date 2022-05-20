import functools
import itertools
import multiprocessing
import operator
import random
import sys
import timeit

import invoke


run = functools.partial(invoke.Context.run, echo=True, pty=True)


@invoke.task
def install(context):
    # type: (invoke.Context) -> None
    run(context, f"{sys.executable} -m pip install --upgrade pip")
    run(context, f"{sys.executable} -m pip install --upgrade -r requirements.txt")


@invoke.task(install)
def clean(context):
    # type: (invoke.Context) -> None
    run(context, f"{sys.executable} setup.py develop --uninstall")
    for artifact in ("*.egg-info", "*.so", "build", "dist"):
        run(context, f"rm -rf {artifact}")
    run(context, f"{sys.executable} -m black .")


@invoke.task(clean)
def build(context):
    # type: (invoke.Context) -> None
    run(context, f"{sys.executable} setup.py develop")


@invoke.task(build)
def test(context):
    # type: (invoke.Context) -> None
    run(context, f"{sys.executable} -m pytest -v")


def do_work(info):
    import automap

    namespace = {"FrozenAutoMap": automap.FrozenAutoMap}
    create_a = timeit.Timer("FrozenAutoMap(keys)", globals=namespace)
    create_d = timeit.Timer("{k: i for i, k in enumerate(keys)}", globals=namespace)
    access_a = timeit.Timer(
        "for key in a: a[key]", "a = FrozenAutoMap(keys)", globals=namespace
    )
    access_d = timeit.Timer(
        "for key in d: d[key]",
        "d = {k: i for i, k in enumerate(keys)}",
        globals=namespace,
    )
    kind, power, factor = info
    items = factor * 10**power
    namespace["keys"] = [kind(_) for _ in range(items)]
    random.shuffle(namespace["keys"])
    iterations = max(create_a.autorange()[0], create_d.autorange()[0])
    create = create_a.timeit(iterations) / create_d.timeit(iterations)
    size = sys.getsizeof(automap.FrozenAutoMap(namespace["keys"])) / sys.getsizeof(
        {k: i for i, k in enumerate(namespace["keys"])}
    )
    iterations = max(access_a.autorange()[0], access_d.autorange()[0])
    access = access_a.timeit(iterations) / access_d.timeit(iterations)
    return items, create, access, size


@invoke.task(test)
def performance(context):
    # type: (invoke.Context) -> None
    print("TYPE\tITEMS\tCREATE\tACCESS\tSIZE")

    def geometric_mean(xs):
        return functools.reduce(operator.mul, xs) ** (1 / len(xs))

    with multiprocessing.get_context("spawn").Pool() as pool:
        for kind in (str, int):
            total_create = []
            total_access = []
            total_size = []
            for items, create, access, size in pool.imap(
                do_work, itertools.product((kind,), range(6), range(1, 10))
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
