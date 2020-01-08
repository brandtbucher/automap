from setuptools import Extension, setup


with open("README.md") as file:
    LONG_DESCRIPTION = file.read()


setup(
    name="automap",
    version="0.1.0",
    description="High-performance autoincremented integer-valued mappings.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=">=3.5.0",
    url="https://github.com/brandtbucher/automap",
    author="Brandt Bucher",
    author_email="brandtbucher@gmail.com",
    license="MIT",
    ext_modules=[Extension("automap", ["automap.c"])],
)
