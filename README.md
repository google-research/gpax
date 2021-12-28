# GPax
A Jax/Flax codebase for Gaussian processes including meta and multi-task Gaussian processes.

The model was originally developed for *[Automatic prior selection for meta Bayesian optimization with a case study on tuning deep neural network optimizers](https://arxiv.org/abs/2109.08215)*, but can be used as example code or building blocks for Gaussian processes in Jax/Flax.

Disclaimer: This is not an officially supported Google product.

## Installation
We recommend using Python 3.7 for stability.

To install the latest development version inside a virtual environment, run
```
python3 -m venv env-pd
source env-pd/bin/activate
pip install --upgrade pip
pip install "git+https://github.com/google/gpax.git#egg=gpax"
```

## Usage
See example colab.

## Citing
```
@article{wang2021automatic,
  title={Automatic prior selection for meta Bayesian optimization with a case study on tuning deep neural network optimizers},
  author={Wang, Zi and Dahl, George E and Swersky, Kevin and Lee, Chansoo and Mariet, Zelda and Nado, Zachary and Gilmer, Justin and Snoek, Jasper and Ghahramani, Zoubin},
  journal={arXiv preprint arXiv:2109.08215},
  year={2021}
}
```
