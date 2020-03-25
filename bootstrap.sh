#!/bin/bash
PYTHON_VERSION="3.8.0"
export MACOSX_DEPLOYMENT_TARGET="10.9"
export PYENV_ROOT="$HOME/.pyenv"
case $OSTYPE in
    linux-*)  # TODO: Just install patchelf in pythonland
        PYENV_REPO="https://github.com/pyenv/pyenv.git"
        PATCHELF_REPO="https://github.com/nixos/patchelf.git"
        PATCHELF_ROOT=".patchelf"
        test -d "$PATCHELF_ROOT" && git -C "$PATCHELF_ROOT" pull || git clone "$PATCHELF_REPO" "$PATCHELF_ROOT" && \
        cd "$PATCHELF_ROOT" && ./bootstrap.sh && ./configure && make && cd -
        export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATCHELF_ROOT/src:$PATH"
        ;;
    msys)
        PYENV_REPO="https://github.com/pyenv-win/pyenv-win.git"
        export PYENV="$PYENV_ROOT/pyenv-win"
        export PATH="$PYENV/bin:$PYENV/shims:$PATH"
        ;;
    *)
        PYENV_REPO="https://github.com/pyenv/pyenv.git"
        export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PYENV_ROOT/libexec:$PATH"
        ;;
esac
test -d "$PYENV_ROOT" && git -C "$PYENV_ROOT" pull || git clone "$PYENV_REPO" "$PYENV_ROOT" && \
test $OSTYPE != msys && eval "$(pyenv init -)" || true && \
test $OSTYPE != msys && pyenv install --skip-existing "$PYTHON_VERSION" || pyenv install --quiet --skip-existing "$PYTHON_VERSION" && \
pyenv global "$PYTHON_VERSION" && pyenv exec python -m venv .envbs && \
test $OSTYPE != msys && source .envbs/bin/activate || source .envbs/Scripts/activate && \
python build_wheels.py
