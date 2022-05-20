import setuptools

with open("README.md") as file:
    LONG_DESCRIPTION = file.read()

setuptools.setup(
    author="Brandt Bucher",
    author_email="brandt@python.org",
    description="High-performance autoincremented integer-valued mappings.",
    ext_modules=[setuptools.Extension("automap", ["automap.c"])],
    license="MIT",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    name="automap",
    python_requires=">=3.7.0",
    url="https://github.com/brandtbucher/automap",
    version="0.5.1",
)
