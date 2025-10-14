# cdr
cosmic density field reconstruction

## Installation
Please make sure that you have installed gcc and mpi library (for example, open-mpi) before installing this library.

If you are a mac user with Apple silicon, the easiest way to go is to install gcc and open-mpi from [homebrew](https://brew.sh/) using the following commands
```bash
brew install gcc@15
HOMEBREW_CC=gcc-15 HOMEBREW_CXX=g++-15 brew install open-mpi --build-from-source
```
Note that, open-mpi has to be built from source with environment variable `HOMEBREW_CC` set to `gcc-15` (surely `gcc-16`, `gcc-17`, ... released in the future should also work). This is because by default homebrew installs a pre-compiled open-mpi library built with Apple Clang, which is not able to build the pfft-python library used by cdr.

Then you can use pip to install this package
```bash
python -m pip install git+https://github.com/zhaoruiyang98/cdr.git
```

Or if you would like to make contributions to this library, it is encouraged to use the modern package manager [uv](https://docs.astral.sh/uv/)
```bash
git clone https://github.com/zhaoruiyang98/cdr.git
cd cdr
uv sync
# or `uv sync --no-dev` if you would like to not install any development dependencies
```
Then you can run `source ./venv/bin/activate` to activate a virtual python environment with all packages installed.