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

"""Linen moduels for Gaussian Process."""

from typing import Any, Callable, Optional, Tuple

from flax.core import frozen_dict
import flax.linen as nn
from gpax import utils
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfed = tfp.experimental.distributions

Array = Any  # Array can be in many different forms (nested, np/jnp).
ArrayMap = Callable[[Array], Array]
# GPRM class is nothing but a container for cholesky decomposition.
CholeskyContainer = tfd.GaussianProcessRegressionModel
Shape = Tuple[int]
Dtype = Any
RNGKey = Any
Initializer = Callable[[RNGKey, Shape, Dtype], Array]


class MaternFiveHalvesKernel(nn.Module):
  """Matern 5/2 kernel."""
  amplitude_init: Initializer = jax.nn.initializers.ones
  length_scales_init: Initializer = jax.nn.initializers.ones

  @nn.compact
  def __call__(
      self, inputs: Array) -> tfp.math.psd_kernels.PositiveSemidefiniteKernel:
    amplitude = self.param('amplitude', self.amplitude_init, [])
    length_scales = self.param('length_scales', self.length_scales_init,
                               inputs.shape[-1])
    return tfp.math.psd_kernels.FeatureScaled(
        tfp.math.psd_kernels.MaternFiveHalves(amplitude=amplitude),
        length_scales)


def _apply_bijector(
    dist: tfd.Distribution,
    bijector: Optional[tfp.bijectors.Bijector]) -> tfd.Distribution:
  """Helper function to optionally apply bijector."""
  if (bijector is None) or (isinstance(bijector, tfp.bijectors.Identity)):
    return dist
  return bijector(dist)


class ConstantFunction(nn.Module):
  constant_init: Initializer = jax.nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    output = self.param('constant', self.constant_init, [])
    return output


class GaussianProcess(nn.Module):
  """Gaussian process.

  Data flow:
    * inputs are fed into `bijector_module` to generate y-value transform.
    * inputs are fed into `feature_module` to generate featurized inputs
    * GP mean function is a linear combination of featurized inputs.
    * featurized inputs are fed into `kernel_module` to generate a kernel
      function (i.e. determines the kernel hyperparameters).

  Attributes:
    kernel_module_gen: Invoked during setup() to create the kernel. Note that it
      returns a kernel Module not a kernel, so that the kernel hyperparameters
      may depend on the input (features).
    bijector_module_gen: Invoked during setup() to optionally create a bijector.
      Bijector is transformation on the random variables modeled by
      GaussianProcess. That is, using tfp.bijectors.Exp() means that GP works in
      the log-transformed space.
    feature_module_gen: Invoked during setup() to optionally create a feature
      mapping. Inputs to this module go through the feature mapping first.
  """
  kernel_module_gen: Callable[[], Callable[
      [Array],
      tfp.math.psd_kernels.PositiveSemidefiniteKernel]] = MaternFiveHalvesKernel
  bijector_module_gen: Callable[[], Callable[
      [Array],
      tfp.bijectors.Bijector]] = lambda: lambda _: tfp.bijectors.Identity()
  feature_module_gen: Callable[[], ArrayMap] = tfp.bijectors.Identity
  mean_fn_module_gen: Callable[[], ArrayMap] = lambda: nn.Dense(1)
  observation_noise_variance_init: Initializer = utils.constant_initializer_factory(
      1e-4)

  def setup(self):
    self.kernel_module = self.kernel_module_gen()
    self.mean_fn = self.mean_fn_module_gen()
    self.bijector_module = self.bijector_module_gen()
    self.feature_module = self.feature_module_gen()
    self.observation_noise_variance = self.param(
        'observation_noise_variance', self.observation_noise_variance_init, [])

  def _gp_and_bijector(
      self,
      inputs: Array) -> Tuple[tfd.GaussianProcess, tfp.bijectors.Bijector]:
    inputs = jnp.asarray(inputs)
    featurized_inputs = self.feature_module(inputs)
    # Invoke mean_fn to initialize the parameters.
    mean_fn = self.mean_fn(featurized_inputs)  # pylint:disable='unused-variable'
    gp = tfd.GaussianProcess(
        self.kernel_module(featurized_inputs),
        featurized_inputs,
        mean_fn=lambda x: jnp.reshape(self.mean_fn(x), [-1]),
        observation_noise_variance=self.observation_noise_variance)
    return gp, self.bijector_module(inputs)

  def __call__(self, inputs: Array) -> tfd.Distribution:
    """Call.

    Let B = batch size, F = number of features

    Args:
      inputs: (B, F) array.

    Returns:
      Distribution with batch shape [], event shape [B].
    """
    gp, bijector = self._gp_and_bijector(inputs)
    return _apply_bijector(gp, bijector)

  def _gp_predictive_and_bijector(
      self,
      inputs: Array,
      cholesky_container: CholeskyContainer,
      include_noise: bool,
  ) -> Tuple[tfd.GaussianProcess, tfp.bijectors.Bijector]:
    bijector = self.bijector_module(inputs)
    featurized_inputs = self.feature_module(inputs)
    dist = cholesky_container.copy(
        index_points=featurized_inputs,
        mean_fn=lambda x: jnp.reshape(self.mean_fn(x), [-1]),
        predictive_noise_variance=None if include_noise else 0.0)
    return dist, bijector

  def predict(self,
              inputs: Array,
              cholesky_container: CholeskyContainer,
              mode: str = 'default',
              *,
              include_noise: bool = True) -> tfd.Distribution:
    """Returns the predictive distribution.

    Let B = batch size, F = number of features.

    Args:
      inputs: (B, F) array.
      cholesky_container: Returned from `cache_cholesky` method.
      mode: 'default', 'latent', or 'independent'.
        'default': Returns the joint distribution over labels at inputs.
        'latent': Same as default, the distribution is defined in transformed
          space, i.e. before applying bijector.
        'independent': Models the labels as independent random variables. The
          independent distribution allows more operations (e.g. quantile()) than
          the joint distribution even when the bijector is not affine.
      include_noise: If True (default), the observation noise is included.

    Returns:
      Distribution over B labels.

    Raises:
      ValueError: If 'mode' has unknown value.
    """
    dist, bijector = self._gp_predictive_and_bijector(inputs,
                                                      cholesky_container,
                                                      include_noise)
    if mode == 'default':
      return _apply_bijector(dist, bijector)
    elif mode == 'latent':
      return dist
    elif mode == 'independent':
      return _apply_bijector(tfd.Normal(dist.mean(), dist.stddev()), bijector)
    else:
      raise ValueError(f'Unknown mode {mode}.')

  def cache_cholesky(self, inputs: Array, ys: Array) -> CholeskyContainer:
    """Returns the posterior predictive distribution.

    Let B = batch size, F = number of features

    Args:
      inputs: (B, F) array.
      ys: (B,) array.

    Returns:
      Use the returned values to initialize a GaussianProcessPredictive module.
    """
    gp, bijector = self._gp_and_bijector(inputs)
    return gp.posterior_predictive(bijector.inverse(ys))

  def nll(self, dataset: utils.Dataset) -> jnp.floating:
    return -jnp.sum(  # pytype: disable=bad-return-type  # jnp-type
        jnp.asarray([
            self(sub_dataset.x).log_prob(sub_dataset.y)
            for sub_dataset in dataset
        ]))

  def train(self,
            dataset: utils.Dataset,
            *,
            params: Optional[frozen_dict.FrozenDict] = frozen_dict.FrozenDict(),
            rng_key: Optional[jax.Array] = None,
            steps: int = 100) -> frozen_dict.FrozenDict:
    """Inference.

    This method optimizes parameters and is NOT designed for apply() usage.
      gp = GaussianProcess()
      gp.train(...) # GOOD
      gp.apply(..., method=gp.train)  # BAD

    Args:
      dataset:
      params: If provided, optimization starts from `params`.
      rng_key: Must be provided if `params` is not provided. Used for
        initializing model parameters.
      steps: number of lbgfs steps.

    Returns:
      Tuned parameters.
    """
    params = params or self.init(rng_key, dataset[0].x)
    tree = utils.ParamsTree(params)

    @jax.jit
    def loss_func(array):
      params = tree.todict(array)
      return self.apply(params, dataset, method=self.nll)

    results = tfp.optimizer.lbfgs_minimize(
        jax.value_and_grad(loss_func),
        initial_position=tree.toarray(params),
        tolerance=1e-6)

    return tree.todict(results.position)


class MultiTaskKernel(nn.Module):
  """Multitask kernel."""
  num_tasks: int
  kernel_module_gen: Callable[[], nn.Module] = MaternFiveHalvesKernel

  @nn.compact
  def __call__(self, inputs: Array) -> (
      tfp.experimental.psd_kernels.MultiTaskKernel):
    """Call.

    Let B = batch size, F = number of features, T = number of tasks.

    Args:
      inputs: (B, F) array.

    Returns:
      PSD kernel over T tasks.
    """
    return tfp.experimental.psd_kernels.Independent(
        self.num_tasks, self.kernel_module_gen()(inputs))


class MultiTaskGaussianProcess(nn.Module):
  """Multitask GP.

  Attributes:
    num_tasks: Number of tasks to be modeled.
    kernel_module_gen: Takes the number of tasks and returns a kernel Module.
      Note that it returns a kernel Module not a kernel, so that the kernel
      hyperparameters may depend on the input (features).
    kernel:
    mean_fn:
  """
  num_tasks: int
  kernel_module_gen: Callable[[int], Callable[
      [Array], tfp.experimental.psd_kernels.MultiTaskKernel]] = MultiTaskKernel

  def setup(self):
    self.kernel_module = self.kernel_module_gen(self.num_tasks)
    self.mean_fn = nn.Dense(self.num_tasks)

  def __call__(self, inputs: Array) -> tfed.MultiTaskGaussianProcess:
    """Call.

    Let B = batch size, F = number of features, T = number of tasks.
    Args:
      inputs: (B, F) array.

    Returns:
      Distribution with batch shape [], event shape [B, T].
    """

    inputs = jnp.asarray(inputs)
    mean_fn = self.mean_fn([[0]]).reshape([-1])
    return tfed.MultiTaskGaussianProcess(
        self.kernel_module(inputs), inputs, mean_fn=lambda _: mean_fn)

  def cache_cholesky(
      self, inputs: Array,
      ys: Array) -> tfed.MultiTaskGaussianProcessRegressionModel:
    """Returns the posterior predictive distribution.

    Let B = batch size, F = number of features, T = number of tasks.

    Args:
      inputs: (B, F) array.
      ys: (B, T) array.

    Returns:
      Distribution with batch shape [], event shape undefined. Use the returned
      distribution to initialize a GaussianProcessPredictive module.
    """
    inputs = jnp.asarray(inputs)
    mean_fn = self.mean_fn([[0]]).reshape([-1])
    return tfed.MultiTaskGaussianProcessRegressionModel.precompute_regression_model(
        self.kernel_module(inputs), inputs, ys, mean_fn=lambda _: mean_fn)
