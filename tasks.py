from functools import reduce
from glob import glob
from itertools import product
from multiprocessing import Pool
from operator import mul
from os import environ, remove, replace
from random import random
from subprocess import run
from sys import abiflags, executable, getsizeof, platform
from sysconfig import get_python_version
from timeit import Timer

from invoke import task


class MockContext:
    @staticmethod
    def run(command: str, echo=False, env=None, replace_env=False) -> None:
        if echo:
            print(command)
        if env is not None:
            env = {**({} if replace_env else environ), **env}
        run(command, env=env, shell=True, check=True)


@task
def clean(context):
    context = MockContext()
    context.run(f"{executable} setup.py develop --uninstall", echo=True)
    for artifact in ("*.egg-info", "*.so", "build", "dist"):
        context.run(f"rm -rf {artifact}", echo=True)
    context.run("black .", echo=True)


@task(clean)
def build(context):
    context = MockContext()
    context.run(f"{executable} -m pip install --upgrade pip", echo=True)
    context.run(f"pip install -r requirements.txt", echo=True)
    context.run(
        f"{executable} setup.py develop sdist bdist_wheel",
        env={"CPPFLAGS": "-Werror -Wno-deprecated-declarations"},
        replace_env=False,
    )

    WHEELS = glob("dist/*.whl")
    assert WHEELS, "No wheels in dist!"
    print("Before:", *WHEELS, sep="\n - ")
    if platform == "linux":
        for so in glob("*.so"):
            context.run(
                f"patchelf --remove-needed libpython{get_python_version()}{abiflags}.so {so}",
                echo=True,
            )
        context.run(f"{executable} setup.py bdist_wheel", echo=True)
        # We're typically eligible for manylinux1... or at least manylinux2010.
        # This will remove the wheel if it was unchanged... but that will cause
        # our assert to fail later, which is what we want!
        for wheel in WHEELS:
            context.run(f"auditwheel repair {wheel} -w dist", echo=True)
            remove(wheel)
    elif platform == "darwin":
        # We lie here, and say our 10.9 64-bit build is a 10.6 32/64-bit one.
        # This is because pip is conservative in what wheels it will use, but
        # Python installations are EXTREMELY liberal in their macOS support.
        # A typical user may be running a 32/64 Python built for 10.6.
        # In reality, we shouldn't worry about supporting 32-bit Snow Leopard.
        for wheel in WHEELS:
            fake = wheel.replace("macosx_10_9_x86_64", "macosx_10_6_intel")
            replace(wheel, fake)
            assert (
                wheel != fake or "TRAVIS" not in environ
            ), "We expected a macOS 10.9 x86_64 build!"
    # Windows is fine.
    FIXED = glob("dist/*.whl")
    print("After:", *FIXED, sep="\n - ")
    assert len(WHEELS) == len(FIXED), "We gained or lost a wheel!"

    context.run("twine check dist/*", echo=True)
    for dist in [".", *glob("dist/*.tar.gz"), *glob("dist/*.whl")]:
        context.run(f"pip install --force-reinstall --no-cache-dir {dist}", echo=True)


@task(build)
def test(context):
    context = MockContext()
    context.run("pytest -v", echo=True)


def do_work(info):
    from automap import FrozenAutoMap

    namespace = {"FrozenAutoMap": FrozenAutoMap}
    create_a = Timer("FrozenAutoMap(keys)", globals=namespace)
    create_d = Timer("{k: i for i, k in enumerate(keys)}", globals=namespace)
    access_a = Timer(
        "for key in a: a[key]", "a = FrozenAutoMap(keys)", globals=namespace
    )
    access_d = Timer(
        "for key in d: d[key]",
        "d = {k: i for i, k in enumerate(keys)}",
        globals=namespace,
    )
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


@task(test)
def performance(context):
    context = MockContext()

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


@task(test)
def release(context):
    context = MockContext()
    context.run("twine upload --skip-existing dist/*", echo=True)
