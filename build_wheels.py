from configparser import ConfigParser
from glob import glob
from os import environ, makedirs
from os.path import dirname
from shutil import copy, rmtree
from subprocess import CalledProcessError, run
from sys import platform
from tempfile import TemporaryDirectory
import venv
from sys import stdout, stderr


# TODO: 32-bit Windows?


LINUX = platform == "linux"
MACOS = platform == "darwin"

PYENV_INSTALL_ARGS = "--skip-existing" if LINUX or MACOS else "--quiet --skip-existing"

WHEELS = {"3.5.4", "3.6.1", "3.7.0", "3.8.0"}
DEFAULTS = {
    "python": "3.8.0",
    "test": "pip install --requirement requirements.txt && pytest",
    "test-if": 'true',
    "upload": "pip install --upgrade twine && twine upload --skip-existing dist/*",
    "upload-if": 'true',
    "wheels": " ".join(sorted(WHEELS)),
    "windows": '32 64' 
}


def _shell(command: str, *, version: str = "", echo: bool = False) -> None:
    if echo:
        print(f"\033[1m{command}\033[0m")
    run(f'pyenv install {PYENV_INSTALL_ARGS} {version or "3.8.0"}', check=True, shell=True)
    # with open(".exec.sh", "w") as file:
    #     file.write(command)
    if LINUX or MACOS:
        run(f"bash -c 'eval \"$(pyenv init -)\" && pyenv shell {version or '3.8.0'} && python -m venv .env{version} && source .env{version}/bin/activate && {command}'", check=True, shell=True)
    else:
        run(f'pyenv global {version or "3.8.0"}', check=True, shell=True)
        run(f'pyenv exec python -m venv .env{version}', check=True, shell=True)
        # makedirs(f'.env{version}/bin', exist_ok=True)
        # try:
        #     copy(f"{dirname(venv.__file__)}/scripts/common/activate", f'.env{version}/bin')
        # except OSError:
        #     pass
        run(f'bash -c "source .env{version}/Scripts/activate && {command}"', check=True, shell=True)
    stderr.flush()
    stdout.flush()


def main() -> None:
    parser = ConfigParser()
    parser.read("setup.cfg")
    config = DEFAULTS.copy()
    if "wheel" in parser:
        config.update(parser["wheel"])
    if unknown := config.keys() - DEFAULTS.keys():
        raise RuntimeError(
            f"Unknown config option{'' if len(unknown) == 1 else 's'}: {'/'.join(sorted(unknown))}! Expected {'/'.join(sorted(DEFAULTS))}."
        )
    wheels = set(config["wheels"].split())
    if not LINUX and not MACOS:
        windows = set(map(int, config['windows'].split()))
        if 64 in windows:
            for wheel in tuple(wheels):
                wheels.add(f'{wheel}-amd64')
        if 32 not in windows:
            for wheel in tuple(wheels):
                if not wheel.endswith('-amd64'):
                    wheels.remove(wheel)
    # if unknown := WHEELS < wheels:
    #     raise RuntimeError(
    #         f"Unknown wheel version{'' if len(unknown) == 1 else 's'}: {'/'.join(sorted(unknown))}! Expected {'/'.join(sorted(WHEELS))}."
    #     )
    # config["wheels"] = [f"{wheel}.0" for wheel in sorted(wheels)]
    config["wheels"] = sorted(wheels)
    rmtree("dist", ignore_errors=True)
    _shell("python -m pip install --upgrade pip")
    _shell("pip install --upgrade setuptools")
    _shell("python setup.py sdist --dist-dir=dist")
    with TemporaryDirectory() as dist:
        for version in config["wheels"]:
            _shell("python -m pip install --upgrade pip", version=version)
            if config["test"]:
                _shell("python setup.py develop", version=version)
                banner = f"Test: Python {version}"
                line = "=" * len(banner)
                print(f"\n{line}\n{banner}\n{line}\n")
                try:
                    _shell(config["test-if"], version=version, echo=True)
                except CalledProcessError:
                    pass
                else:
                    _shell(config["test"], version=version, echo=True)
                finally:
                    print(f"\n{line}\n")
                    _shell("python setup.py develop --uninstall", version=version)
            _shell("pip install --upgrade setuptools wheel", version=version)
            if MACOS:
                # We lie for macOS, and say our 10.9 64-bit build is a 10.6
                # 32/64-bit one. This is because pip is conservative in what
                # wheels it will use, but Python installations are EXTREMELY
                # liberal in their macOS support. A typical user may be running
                # a 32/64 Python built for 10.6. In reality, we shouldn't worry
                # about supporting 32-bit Snow Leopard.
                _shell(
                    f"python setup.py bdist_wheel --dist-dir={dist} --plat-name=macosx_10_6_intel", version=version
                )
            elif LINUX:
                _shell(f"python setup.py bdist_wheel --dist-dir={dist}", version=version)
            else:
                _shell(f"python setup.py bdist_wheel --dist-dir=dist", version=version)
        wheels = glob(f"{dist}/*")
        if LINUX:
            _shell("pip install --upgrade auditwheel")
            for wheel in wheels:
                _shell(f"auditwheel repair --wheel-dir=dist {wheel}")
        elif MACOS:
            _shell("pip install --upgrade delocate")
            for wheel in wheels:
                _shell(f"delocate-wheel --wheel-dir=dist --require-archs=intel {wheel}")
        # else:
        #     for wheel in wheels:
        #         copy(wheel, "dist")
    if config["upload"]:
        banner = "Upload"
        line = "=" * len(banner)
        print(f"\n{line}\n{banner}\n{line}\n")
        try:
            _shell(config["upload-if"], version=config["python"], echo=True)
        except CalledProcessError:
            pass
        else:
            _shell(config["upload"], version=config["python"], echo=True)
        finally:
            print(f"\n{line}\n")
            # _shell("python setup.py develop --uninstall", version=config["python"], )

    print(*sorted(glob("dist/*")), sep="\n")  # XXX


if __name__ == "__main__":
    main()
