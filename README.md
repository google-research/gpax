# GPax
A codebase for Gaussian processes in Jax. 

Disclaimer: This is not an officially supported Google product.

## Gaussian process probes (GPP)
Please find algorithm descriptions in *[Gaussian Process Probes (GPP) for Uncertainty-Aware Probing](https://arxiv.org/abs/2305.18213)*.

To use GPP, simply call [`gpp`](https://github.com/google-research/gpax/blob/main/GPax/probing/probabilistic_probe.py#L25) with embeddings of queries and observed data. 

To obtain uncertainty measurements of a Beta GP, call [`beta_gp_uncertainty`](https://github.com/google-research/gpax/blob/1ac63b62a883099517731bb26a488b064a700998/GPax/probing/gp.py) with predictions returned by [`beta_gp_predict`](https://github.com/google-research/gpax/blob/1ac63b62a883099517731bb26a488b064a700998/GPax/probing/gp.py).

Baseline probabilistic probing methods are also implemented in this codebase, including [linear probe ensembles](https://github.com/google-research/gpax/blob/main/GPax/probing/probabilistic_probe.py#L68) and [Gaussian process regression](https://github.com/google-research/gpax/blob/main/GPax/probing/probabilistic_probe.py#L46).

For out-of-distribution (OOD) detection, use [`latent_var`](https://github.com/google-research/gpax/blob/1ac63b62a883099517731bb26a488b064a700998/GPax/probing/gp.py#LL463C8-L463C18), the variance of the latent function returned by [`gpp`](https://github.com/google-research/gpax/blob/main/GPax/probing/probabilistic_probe.py#L25). 

Other score-based OOD detection methods implemented in this codebase are [Mahalanobis distance](https://github.com/google-research/gpax/blob/1ac63b62a883099517731bb26a488b064a700998/GPax/probing/probabilistic_probe.py#L96) and [maximum predicted softmax probabilities](https://github.com/google-research/gpax/blob/1ac63b62a883099517731bb26a488b064a700998/GPax/probing/probabilistic_probe.py#L90).

### Citation
```
@article{wang2023gpp,
  title={{Gaussian Process Probes (GPP) for Uncertainty-Aware Probing}},
  author={Zi Wang and
          Alexander Ku and
          Jason Baldridge and
          Thomas L Griffiths and
          Been Kim},
  journal={arXiv preprint arXiv:2305.18213},
  year={2023}
}
```

## Pre-trained Gaussian processes

Please find algorithm descriptions in *[Pre-trained Gaussian processes for Bayesian optimization](https://arxiv.org/abs/2109.08215)*. An alternative implementation can be found at https://github.com/google-research/hyperbo.

Implemented models include vanilla Gaussian processes ([`GaussianProcess`](https://github.com/google-research/gpax/blob/main/GPax/models/gp.py#L74)) as well as meta and multi-task Gaussian processes ([`MultiTaskGaussianProcess`](https://github.com/google-research/gpax/blob/main/GPax/models/gp.py#L279)). 

For pre-training the multi-task Gaussian process, you can call an optimizer (minimization) on the [empirical KL divergence (EKL) objective](https://github.com/google-research/gpax/blob/main/GPax/objectives/empirical_kl_divergence.py) or the [negative log likelihood (NLL) objective](https://github.com/google-research/gpax/blob/main/GPax/objectives/neg_log_likelihood.py). Examples of evaluating these objectives can be found in [the test for EKL](https://github.com/google-research/gpax/blob/main/GPax/objectives/empirical_kl_divergence_test.py) and [the test for NLL](https://github.com/google-research/gpax/blob/main/GPax/objectives/neg_log_likelihood_test.py).

We also implemented [classic acquisition functions](https://github.com/google-research/gpax/blob/main/GPax/bayesopt/acquisitions.py) for Bayesian optimization. See [`GPax/bayesopt/acquisitions_test.py`](https://github.com/google-research/gpax/blob/main/GPax/bayesopt/acquisitions_test.py) for an example of how to evaluate these acquisition functions.

### Citation
```
@article{wang2023hyperbo,
  title={{Pre-trained Gaussian processes for Bayesian optimization}},
  author={Zi Wang and
          George E. Dahl and
          Kevin Swersky and
          Chansoo Lee and
          Zachary Nado and
          Justin Gilmer and
          Jasper Snoek and
          Zoubin Ghahramani},
  journal={arXiv preprint arXiv:2109.08215},
  year={2023}
}
```

## Installation
We recommend using Python 3.7 for stability.

To install the latest development version inside a virtual environment, run
```
python3 -m venv env-pd
source env-pd/bin/activate
pip install --upgrade pip
pip install "git+https://github.com/google-research/gpax.git#egg=gpax"
```


