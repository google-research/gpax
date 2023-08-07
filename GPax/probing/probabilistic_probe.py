# coding=utf-8
# Copyright 2023 GPax Authors.
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

"""Measure uncertainty for GPP, GPR, linear ensemble."""

from gpax.probing import gp
import jax
import numpy as np
import sklearn.linear_model as sklm


@jax.jit
def gpp(x_query, x_observed=None, y_observed=None, alpha_eps=0.1, strength=5.0):
  """GPP."""
  mean_func = gp.constant_mean
  cov_func = gp.cosine_kernel
  params = {
      'alpha_eps': alpha_eps,
      'strength': strength,
  }
  predictions = gp.beta_gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      x_query=x_query,
      x_observed=x_observed,
      y_observed=y_observed[:, None] if y_observed is not None else None,
      params=params,
  )
  measures = gp.beta_gp_uncertainty(predictions)
  return jax.tree_map(lambda x: x[:, 0], measures)


@jax.jit
def gpr(x_query, x_observed=None, y_observed=None):
  """GPR."""
  mean_func = gp.constant_mean
  cov_func = gp.cosine_kernel
  params = {
      'constant': 0.0,
      'noise_variance': 0.1,
      'scale': 6.0,
      'signal_variance': 6.0,
  }
  mu, var = gp.gpr_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      x_query=x_query,
      x_observed=x_observed,
      y_observed=y_observed[:, None] if y_observed is not None else None,
      params=params,
  )
  measures = gp.gp_uncertainty(mu, var)
  return jax.tree_map(lambda x: x[:, 0], measures)


def lpe(x_query, x_observed=None, y_observed=None, repeats=int(1e2)):
  """Linear probe ensemble."""
  p_samples = []
  pos_idx = np.where(y_observed)[0]
  neg_idx = np.where(y_observed == 0)[0]
  if pos_idx.shape[0] == 0 or neg_idx.shape[0] == 0:
    raise ValueError('Must have at least 1 positive and 1 negative examples.')
  n = len(x_observed) - 2
  for _ in range(repeats):
    idx_0 = np.random.choice(pos_idx)
    idx_1 = np.random.choice(neg_idx)
    idx = np.random.choice(np.arange(n + 2), (n,))
    idx = np.hstack(([idx_0, idx_1], idx))
    # make sure to select at least 1 positive and 1 negative
    cls = sklm.LogisticRegression().fit(x_observed[idx], y_observed[idx])
    cls_p = cls.predict_proba(x_query)[:, 1]
    p_samples.append(cls_p)
  p_samples = np.array(p_samples).T  # num_inputs x n
  measures = gp.classifier_samples_uncertainty(p_samples, True)
  return jax.tree_map(lambda x: x[:, 0], measures)


def lp_maxprob(x_query, x_observed=None, y_observed=None):
  cls = sklm.LogisticRegression().fit(x_observed, y_observed)
  cls_p = cls.predict_proba(x_query)[:, 1]
  return {'episteme': np.max([cls_p, 1 - cls_p], axis=0)}


def maha(x_query, x_observed=None, y_observed=None):
  """Mahalanobis score https://arxiv.org/pdf/1807.03888.pdf."""
  x0 = x_observed[np.where(y_observed == 0)[0]]
  x1 = x_observed[np.where(y_observed)[0]]
  if x0.shape[0] == 0 or x1.shape[0] == 0:
    raise ValueError('Must have at least 1 positive and 1 negative examples.')
  if x0.shape[0] == 1 and x1.shape[0] == 1:
    return {'episteme': np.zeros(len(x_query))}
  mu0 = np.mean(x0, axis=0)
  cov0 = np.cov(x0.T, bias=True)
  mu1 = np.mean(x1, axis=0)
  cov1 = np.cov(x1.T, bias=True)
  cov = cov0 + cov1
  if not cov.shape:
    cov = [[cov]]
  cov = np.linalg.pinv(cov)
  delta0 = x_query - mu0
  delta1 = x_query - mu1
  dist0 = np.sum(np.dot(delta0, cov) * delta0, axis=1)
  dist1 = np.sum(np.dot(delta1, cov) * delta1, axis=1)
  return {'episteme': -np.min([dist0, dist1], axis=0)}


def deep_nearest_neighbor(x_query, x_observed=None, y_observed=None):
  """Deep nearest neighbor score https://arxiv.org/pdf/2204.06507.pdf."""
  del y_observed
  k = len(x_query) // 2
  delta = x_query / np.linalg.norm(
      x_query, axis=1, keepdims=True
  ) - x_observed / np.linalg.norm(x_observed, axis=1, keepdims=True)
  dist = np.linalg.norm(delta, axis=1)
  sorted_dist = np.sort(dist)
  return {'episteme': -sorted_dist[k]}
