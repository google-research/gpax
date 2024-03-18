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

"""Tests for gp."""

import functools

from absl.testing import absltest
from flax import linen as nn
from gpax import utils
from gpax.models import gp
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp


class GaussianProcessTest(absltest.TestCase):

  def test_fixed_inits(self):

    key1, key2, key3 = random.split(random.PRNGKey(1), 3)
    xx = random.uniform(key1, (8, 5))
    yy = random.uniform(key2, (8,))
    kernel = functools.partial(
        gp.MaternFiveHalvesKernel,
        amplitude_init=utils.constant_initializer_factory(1.),
        length_scales_init=utils.constant_initializer_factory(1.))
    mean_fn = functools.partial(
        gp.ConstantFunction,
        constant_init=utils.constant_initializer_factory(5.1))

    class EightFeatures(nn.Module):

      @nn.compact
      def __call__(self, x):
        return nn.tanh(nn.Dense(8)(x))

    model = gp.GaussianProcess(
        kernel_module_gen=kernel,
        mean_fn_module_gen=mean_fn,
        feature_module_gen=EightFeatures,
        observation_noise_variance_init=utils.constant_initializer_factory(
            np.exp(-4)))
    params = model.init(key3, xx)

    @jax.jit
    def nll(params):
      return model.apply(params, [utils.SubDataset(xx, yy)], method=model.nll)

    nll_before_train = nll(params)
    params = model.train([utils.SubDataset(xx, yy)], params=params, steps=5)
    nll_after_train = nll(params)
    self.assertLess(nll_after_train, nll_before_train)

  def test_shapes(self):

    key1, key2 = random.split(random.PRNGKey(0), 2)
    xx = random.uniform(key1, (8, 5))
    yy = random.uniform(key1, (8,))
    model_kwargs = {'feature_module_gen': lambda: nn.Dense(6)}
    model = gp.GaussianProcess(**model_kwargs)
    params = model.init(key2, xx)

    # Prior distribution.
    dist = model.apply(params, xx)
    self.assertEmpty(dist.batch_shape)
    self.assertEqual(dist.event_shape, yy.shape)
    self.assertEmpty(dist.log_prob(yy).shape)

    # Take the posterior distribution.
    xx2 = random.uniform(key1, (10, 5))
    cached_cholesky = model.apply(params, xx, yy, method=model.cache_cholesky)
    posterior = model.apply(params, xx2, cached_cholesky, method=model.predict)

    self.assertEqual(posterior.batch_shape, [])
    self.assertEqual(posterior.event_shape, [10])
    self.assertSequenceEqual(posterior.mean().shape, [10])
    self.assertSequenceEqual(posterior.covariance().shape, [10, 10])

    # Take the posterior distribution, without the noise.
    posterior_without_noise = model.apply(
        params, xx2, cached_cholesky, method=model.predict, include_noise=False)
    diff = posterior.covariance() - posterior_without_noise.covariance()
    # The difference in the covariance matrix should be identity times
    # noise variance.
    np.testing.assert_array_almost_equal(
        diff / diff[0, 0], jnp.eye(10), decimal=3)

    return dist, posterior

  def test_nonaffine_bijector(self):
    key1, key2 = random.split(random.PRNGKey(0), 2)
    xx = random.uniform(key1, (8, 5))
    yy = random.uniform(key1, (8,))
    model = gp.GaussianProcess(
        bijector_module_gen=lambda: lambda _: tfp.bijectors.Exp())
    params = model.init(key2, xx)

    # Set up for posterior inference.
    xx2 = random.uniform(key1, (10, 5))
    cached_cholesky = model.apply(params, xx, yy, method=model.cache_cholesky)

    # Get the joint posterior distribution.
    posterior = model.apply(params, xx2, cached_cholesky, method=model.predict)
    self.assertEqual(posterior.batch_shape, [])
    self.assertEqual(posterior.event_shape, [10])
    self.assertSequenceEqual(posterior.sample(seed=key2).shape, [10])
    # Other than sampling, we cannot do much with the posterior when we
    # use a non-affine transformation.
    with self.assertRaises(NotImplementedError):
      posterior.mean()
    with self.assertRaises(NotImplementedError):
      posterior.covariance()
    with self.assertRaises(NotImplementedError):
      posterior.quantile(.7)

    # Get the joint posterior distribution in latent space.
    posterior = model.apply(
        params, xx2, cached_cholesky, 'latent', method=model.predict)
    self.assertEqual(posterior.batch_shape, [])
    self.assertEqual(posterior.event_shape, [10])
    # We can compute mean and covariance.
    self.assertSequenceEqual(posterior.mean().shape, [10])
    self.assertSequenceEqual(posterior.covariance().shape, [10, 10])
    with self.assertRaises(NotImplementedError):
      posterior.quantile(.7)

    # Get pointwise posterior distribution.
    posterior = model.apply(
        params, xx2, cached_cholesky, 'independent', method=model.predict)
    # Posterior is no longer a distribution on vector; instead, it is
    # a collection of scalar distributions.
    self.assertEqual(posterior.batch_shape, [10])
    self.assertEqual(posterior.event_shape, [])
    self.assertSequenceEqual(posterior.sample(seed=key2).shape, [10])
    # We can't take mean, because bijector is not affine
    with self.assertRaises(NotImplementedError):
      posterior.mean()
    # But we can take quantiles.
    self.assertSequenceEqual(posterior.quantile(.7).shape, [10])

  def test_inference(self):
    key1 = random.PRNGKey(0)
    key1_replica = random.PRNGKey(0)
    dataset = [
        utils.SubDataset(
            random.uniform(key1, (8, 5)), random.uniform(key1, (8,))),
        utils.SubDataset(
            random.uniform(key1, (8, 5)), random.uniform(key1, (8,)))
    ]

    model = gp.GaussianProcess()
    params = model.init(key1, dataset[0].x)
    nll_before = model.apply(params, dataset, method=model.nll)

    params = model.train(dataset, rng_key=key1_replica, steps=5)
    nll_after = model.apply(params, dataset, method=model.nll)
    self.assertLessEqual(nll_after, nll_before)

    dist = model.apply(params, dataset[0].x)
    self.assertEmpty(dist.batch_shape)
    self.assertEqual(dist.event_shape, dataset[0].y.shape)
    self.assertEmpty(dist.log_prob(dataset[0].y).shape)


class MultiTaskGaussianProcessTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.key1, self.key2 = random.split(random.PRNGKey(0), 2)
    self.xx = random.uniform(self.key1, (8, 5))
    self.yy = random.uniform(self.key1, (8, 3))

  def test_prior_shapes(self):
    xx, yy = self.xx, self.yy
    model = gp.MultiTaskGaussianProcess(3)
    params = model.init(self.key2, xx)

    # Prior distribution.
    dist = model.apply(params, xx)
    self.assertEmpty(dist.batch_shape)
    self.assertEqual(dist.event_shape, yy.shape)
    self.assertTrue(jnp.isfinite(dist.log_prob(yy)))

  def test_posterior_shapes(self):
    xx, yy = self.xx, self.yy
    model = gp.MultiTaskGaussianProcess(3)
    params = model.init(self.key2, xx)

    # Take the posterior distribution.
    xx2 = random.uniform(self.key1, (10, 5))
    gprm = model.apply(params, xx, yy, method=model.cache_cholesky)
    posterior = gprm.copy(index_points=xx2)
    # self.assertEqual(posterior.batch_shape, [])
    self.assertEqual(posterior.event_shape, (10, 3))


if __name__ == '__main__':
  absltest.main()
