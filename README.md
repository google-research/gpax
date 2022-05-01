# GPax
A Jax/Flax codebase for Gaussian processes including meta and multi-task Gaussian processes.

The model was originally developed for *[Pre-training helps Bayesian optimization too](https://ziw.mit.edu/pub/hyperbo.pdf)*, but can be used as example code or building blocks for Gaussian processes in Jax/Flax.

Disclaimer: This is not an officially supported Google product.

## Installation
We recommend using Python 3.7 for stability.

To install the latest development version inside a virtual environment, run
```
python3 -m venv env-pd
source env-pd/bin/activate
pip install --upgrade pip
pip install "git+https://github.com/google-research/gpax.git#egg=gpax"
```

## Usage
See tests.

## Citing
```
@article{wang2021hyperbo,
  title={Pre-training helps Bayesian optimization too},
  author={Wang, Zi and Dahl, George E and Swersky, Kevin and Lee, Chansoo and Mariet, Zelda and Nado, Zachary and Gilmer, Justin and Snoek, Jasper and Ghahramani, Zoubin},
  journal={arXiv preprint arXiv:2109.08215},
  year={2022}
}
```
