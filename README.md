# CAMP

CAMP (Cartesian Atomistic Moment Potentials) for materials and molecular systems.

## Installation

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

which will install the dependencies with pinned versions.

## Examples

Example training scripts and configuration files can be found in the [scripts](./scripts) directory.

Pretrained models for LiPS, md17, bulk water, and bilayer graphene are available at: https://github.com/wengroup/camp_run.

## Citation

Wen, M., Huang, W. F., Dai, J., & Adhikari, S. (2024). Cartesian Atomic Moment Machine Learning Interatomic Potentials. arXiv preprint arXiv:2411.12096.

```latex
@article{wen2025cartesian,
	author  = {Wen, Mingjian and Huang, Wei-Fan and Dai, Jin and Adhikari, Santosh},
	title   = {Cartesian atomic moment machine learning interatomic potentials},
	journal = {npj Computational Materials},
	volume  = {11},
	number  = {128},
	year    = {2025},
	doi     = {10.1038/s41524-025-01623-4}
}
```
