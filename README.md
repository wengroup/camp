# CAMP

CAMP (Cartesian Atomistic Moment Potential) is an equivariant graph neural network interatomic potential.



## Install

Follow the official documentation to install [pytorch>=2.0.0](https://pytorch.org/get-started/locally/).

Then get CAMP and install it by:

```sh
git clone https://github.com/mjwen/camp.git
cd camp
pip install -e .
```

If you get into trouble with some of the dependencies, try

```sh
pip install -e '.[strict]'
```
which will install the dependencies with specific versions.
