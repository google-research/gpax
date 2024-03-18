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

"""Empirical KL divergence.

KL divergence between empirical estimates and modelpredictions.
"""

import logging

from flax.core import frozen_dict
from gpax import utils
from gpax.models import gp
import jax
from jax import numpy as jnp
from jax.custom_derivatives import custom_vjp
import jax.scipy.linalg as jspla

vmap = jax.vmap


def cholesky_cache(spd_matrix, cached_cholesky):
  """Computes the Cholesky factor of `spd_matrix` unless one is already given."""
  if cached_cholesky is not None:
    chol_factor = cached_cholesky
  else:
    chol_factor = jspla.cholesky(spd_matrix, lower=True)

  return chol_factor


@custom_vjp
def inverse_spdmatrix_vector_product(spd_matrix, x, cached_cholesky=None):
  """Computes the inverse matrix vector product where the matrix is SPD."""
  chol_factor = cholesky_cache(spd_matrix, cached_cholesky)

  out = jspla.cho_solve((chol_factor, True), x)
  return out


def kl_multivariate_normal(mu0,
                           cov0,
                           mu1,
                           cov1,
                           weight=1.,
                           partial=True,
                           feat0=None,
                           eps=0.):
  """Computes KL divergence between two multivariate normal distributions.

  Args:
    mu0: mean for the first multivariate normal distribution.
    cov0: covariance matrix for the first multivariate normal distribution.
    mu1: mean for the second multivariate normal distribution.
    cov1: covariance matrix for the second multivariate normal distribution.
      cov1 must be invertible.
    weight: weight for the returned KL divergence.
    partial: only compute terms in KL involving mu1 and cov1 if True.
    feat0: (optional) feature used to compute cov0 if cov0 = feat0 * feat0.T /
      feat0.shape[1]. For a low-rank cov0, we may have to compute the KL
      divergence for a degenerate multivariate normal.
    eps: (optional) small positive value added to the diagonal terms of cov0 and
      cov1 to make them well behaved.

  Returns:
    KL divergence. The returned value does not include terms that are not
    affected by potential model parameters in mu1 or cov1.
  """
  if not cov0.shape:
    cov0 = cov0[jnp.newaxis, jnp.newaxis]
  if not cov1.shape:
    cov1 = cov1[jnp.newaxis, jnp.newaxis]

  if eps > 0.:
    cov0 = cov0 + jnp.eye(cov0.shape[0]) * eps
    cov1 = cov1 + jnp.eye(cov1.shape[0]) * eps

  mu_diff = mu1 - mu0
  chol1 = jspla.cholesky(cov1, lower=True)
  cov1invmudiff = inverse_spdmatrix_vector_product(
      cov1, mu_diff, cached_cholesky=chol1)
  # pylint: disable=g-long-lambda
  func = lambda x: inverse_spdmatrix_vector_product(
      cov1, x, cached_cholesky=chol1)
  trcov1invcov0 = jnp.trace(vmap(func)(cov0))
  mahalanobis = jnp.dot(mu_diff, cov1invmudiff)
  logdetcov1 = jnp.sum(2 * jnp.log(jnp.diag(chol1)))
  common_terms = trcov1invcov0 + mahalanobis + logdetcov1
  if partial:
    return 0.5 * weight * common_terms
  else:
    if feat0 is not None and feat0.shape[0] > feat0.shape[1]:
      logging.info('Using pseudo determinant of cov0.')
      sign, logdetcov0 = jnp.linalg.slogdet(
          jnp.divide(jnp.dot(feat0.T, feat0), feat0.shape[1]))
      logging.info(msg=f'Pseudo logdetcov0 = {logdetcov0}')
      assert sign == 1., 'Pseudo determinant of cov0 is 0 or negative.'

      # cov0inv is computed for more accurate pseudo KL. feat0 may be low rank.
      cov0inv = jnp.linalg.pinv(cov0)
      return 0.5 * weight * (
          common_terms - logdetcov0 -
          jnp.linalg.matrix_rank(jnp.dot(cov0inv, cov0)) + jnp.log(2 * jnp.pi) *
          (cov1.shape[0] - feat0.shape[1]))
    else:
      sign, logdetcov0 = jnp.linalg.slogdet(cov0)
      logging.info(msg=f'sign = {sign}; logdetcov0 = {logdetcov0}')
      assert sign == 1., 'Determinant of cov0 is 0 or negative.'
      return 0.5 * weight * (common_terms - logdetcov0 - cov0.shape[0])


def objective(model: gp.GaussianProcess,
              params: frozen_dict.FrozenDict,
              dataset: utils.Dataset,
              partial: bool = True):
  """Compute empirical KL divergence of model to empirical estimates.

  The returned regularizer aims to minimize the distance between the
  multivariate normal specified by sample mean/covariance and the multivariate
  normal specified by the parameterized GP. We support KL divergence as distance
  or squared Euclidean distance.

  Args:
    model: gp.GaussianProcess.
    params: model parameters.
    dataset: a list of SubDataset. For aligned sub-dataset, this function
      should only be used if each aligned sub-dataset only has (?, m) for
      y shape, where m > 1.
    partial: set to True if only compute the partial KL divergence that is only
      relevant to model prediction.

  Returns:
    KL divergence between empirical estimates and model predictions.
  """

  def kl_per_subset(sub_dataset):
    """Compute the regularizer on a subset of dataset keys."""
    if sub_dataset.y.shape[0] == 0:
      return 0.
    mu_data = jnp.mean(sub_dataset.y, axis=1)
    cov_data = jnp.cov(sub_dataset.y, bias=True)
    pred = model.apply(params, sub_dataset.x)

    return kl_multivariate_normal(
        mu0=mu_data,
        cov0=cov_data,
        mu1=pred.mean(),
        cov1=pred.covariance(),
        partial=partial,
        feat0=sub_dataset.y - mu_data[:, None])

  return jnp.sum(
      jnp.array([
          kl_per_subset(sub_dataset)
          for sub_dataset in dataset
          if sub_dataset.aligned is not None
      ]))
