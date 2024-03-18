# coding=utf-8
# Copyright 2024 GPax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Acquisition functions for Bayesian optimization."""

import abc
import enum
from typing import Any

import jax  # pylint:disable=unused-import
from tensorflow_probability.substrates import jax as tfp

Array = Any  # Array can be in many different forms (nested, np/jnp).
tfd = tfp.distributions


class Goal(enum.Enum):
  MAXIMIZE = 'MAXIMIZE'
  MINIMIZE = 'MINIMIZE'


class AcquisitionFunction(abc.ABC):

  @abc.abstractmethod
  def evaluate(self, posterior: tfd.Distribution) -> Array:
    """Evaluates the acquisition function."""
    pass


class Quantile(AcquisitionFunction):
  """Take the p-th quantile of the posterior.

  UpperConfidenceBound(b) is the same as Quantile(tfd.Normal(0,1).cdf(b))
  for normally distributed posteriors. Quantile can be computed on
  non-linear transfomration of the Normal distribution, while
  `UpperConfidenceBound` cannot.
  """

  def __init__(self, quantile: Array):
    """Init.

    Args:
      quantile: Array of quantiles.
    """
    self.quantile = quantile

  def evaluate(self, posterior: tfd.Distribution) -> Array:
    """Evaluate.

    Args:
      posterior: Distribution of batch shape B and event shape []. The
        distribution must support quantile().

    Returns:
      Array of shape `concatenate([self.quantile.shape, B])`.
    """
    return posterior.quantile(self.quantile)


class UpperConfidenceBound(AcquisitionFunction):
  """UCB Acquisition function.

  Quantile(q) is the same as UpperConfidenceBound(tfd.Normal(0,1).quantile(q))
  for normally distributed posteriors. But `UpperConfidenceBound` can be
  computed even if posterior has covariance matrix, while `Quantile` cannot.
  """

  def __init__(self, beta: Array):
    """Init.

    Args:
      beta: Array of UCB coefficients. In a typical use case, this would be a
        scalar.
    """
    self.beta = beta

  def evaluate(self, posterior: tfd.Distribution) -> Array:
    """Evaluate.

    Args:
      posterior: Distribution of batch shape B and event shape E. Batch shape B
        must be broadcastible with `self.beta.shape`. The distribution must
        support stddev().

    Returns:
      Array of shape `concatenate([B, E])`.
    """
    return posterior.mean() + posterior.stddev() * self.beta


class ImprovementZScore(AcquisitionFunction):
  """Computes the Z score of the improvement."""

  def __init__(self, target: Array, goal: Goal):
    """Init.

    Args:
      target: Target value to compute improvement over. In a typical use case,
        this would be a scalar.
      goal: Acquisition value is non-negative and should be maximized if goal is
        MAXIMIZE, and non-positive and should be minimized if goal is MINIMIZE.
    """
    self.target = target
    self.goal = goal

  def evaluate(self, posterior: tfd.Distribution) -> Array:
    multipler = 1.0 if self.goal == Goal.MAXIMIZE else -1.0
    return multipler * (self.target - posterior.mean()) / posterior.stddev()


class ExpectedImprovement(AcquisitionFunction):
  """Expected Improvement."""

  def __init__(self, target: Array, goal: Goal):
    """Init.

    Args:
      target: Target value to compute improvement over. In a typical use case,
        this would be a scalar.
      goal: Acquisition value is non-negative and should be maximized if goal is
        MAXIMIZE, and non-positive and should be minimized if goal is MINIMIZE.
    """
    self.target = target
    self.goal = goal

  def evaluate(self, posterior: tfd.Distribution) -> Array:
    gamma = ImprovementZScore(self.target, self.goal).evaluate(posterior)
    normal = tfd.Normal(0., 1.)
    return (normal.prob(gamma) - gamma *
            (1 - normal.cdf(gamma))) * posterior.stddev()


class ThomsonSampling(AcquisitionFunction):
  """Samples from posterior."""

  def __init__(self, seed: Array):
    """Init.

    Args:
      seed: Random seed such as jax.random.PRNGKey to be used for sampling.
    """
    self.seed = seed

  def evaluate(self, posterior: tfd.Distribution) -> Array:
    return posterior.sample(seed=self.seed)
