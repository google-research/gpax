# GPax
A codebase for Gaussian processes in Jax. Implemented models include vanilla Gaussian processes as well as meta and multi-task Gaussian processes.

The model was originally developed for *[Pre-trained Gaussian processes for Bayesian optimization](https://arxiv.org/pdf/2109.08215.pdf)*, but can be used as example code or building blocks for Gaussian processes in Jax/Flax and Tensorflow Probability.

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
@article{wang2023hyperbo,
  title={Pre-trained Gaussian processes for Bayesian optimization},
  author={Wang, Zi and Dahl, George E and Swersky, Kevin and Lee, Chansoo and Mariet, Zelda and Nado, Zachary and Gilmer, Justin and Snoek, Jasper and Ghahramani, Zoubin},
  journal={arXiv preprint arXiv:2109.08215},
  year={2023}
}
```
