import setuptools
import typing as tp
import site
import os

with open("README.md") as file:
    LONG_DESCRIPTION = file.read()


def get_ext_dir(*components: tp.Iterable[str]) -> tp.Sequence[str]:
    dirs = []
    for sp in site.getsitepackages():
        fp = os.path.join(sp, *components)
        if os.path.exists(fp):
            dirs.append(fp)
    return dirs


extension = setuptools.Extension(
    "automap",
    ["automap.c"],
    include_dirs=get_ext_dir("numpy", "core", "include"),
    library_dirs=get_ext_dir("numpy", "core", "lib"),
    libraries=["npymath"],  # not including mlib at this time
)


setuptools.setup(
    author="Brandt Bucher",
    author_email="brandt@python.org",
    description="High-performance autoincremented integer-valued mappings.",
    ext_modules=[extension],
    license="MIT",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    name="automap",
    python_requires=">=3.7.0",
    url="https://github.com/brandtbucher/automap",
    version="0.6.2",
)
