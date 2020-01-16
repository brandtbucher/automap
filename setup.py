from setuptools import Extension, setup


with open("README.md") as file:
    LONG_DESCRIPTION = file.read()


setup(
    author="Brandt Bucher",
    author_email="brandtbucher@gmail.com",
    description="High-performance autoincremented integer-valued mappings.",
    ext_modules=[Extension("automap", ["automap.c"])],
    license="MIT",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    name="automap",
    python_requires=">=3.6.0",
    url="https://github.com/brandtbucher/automap",
    version="0.4.1",
)
