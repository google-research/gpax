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

"""Implementations of Gaussian processes for classification."""
import functools
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.linalg as jspla

vmap = jax.vmap


def _verify_params(model_params, expected_keys):
  """Verify that dictionary params has the expected keys."""
  if not set(expected_keys).issubset(set(model_params.keys())):
    raise ValueError(
        f'Expected parameters are {sorted(expected_keys)}, '
        f'but received {sorted(model_params.keys())}.'
    )


def retrieve_params(params, keys, warp_func):
  """Returns a list of parameter values (warped if specified) by keys' order."""
  _verify_params(params, keys)
  if warp_func:
    values = [
        warp_func[key](params[key]) if key in warp_func else params[key]
        for key in keys
    ]
  else:
    values = [params[key] for key in keys]
  return values


def constant_mean(params, x, warp_func=None):
  """Constant mean function."""
  (val,) = retrieve_params(params, ['constant'], warp_func)
  return jnp.full((x.shape[0], 1), val)


def covariance_matrix(kernel):
  """Decorator to kernels to obtain the covariance matrix."""

  @functools.wraps(kernel)
  def matrix_map(params, vx1, vx2=None, warp_func=None, diag=False):
    """Returns the kernel matrix of input array vx1 and input array vx2.

    Args:
      params: parameters for the kernel.
      vx1: n1 x d dimensional input array representing n1 data points.
      vx2: n2 x d dimensional input array representing n2 data points. If it is
        not specified, vx2 is set to be the same as vx1.
      warp_func: optional dictionary that specifies the warping function for
        each parameter.
      diag: flag for returning diagonal terms of the matrix (True) or the full
        matrix (False).

    Returns:
      The n1 x n2 dimensional covariance matrix derived from kernel evaluations
        on every pair of inputs from vx1 and vx2 by default. If diag=True and
        vx2=None, it returns the diagonal terms of the n1 x n1 covariance
        matrix.
    """
    cov_func = functools.partial(kernel, params, warp_func=warp_func)
    mmap = vmap(lambda x: vmap(lambda y: cov_func(x, y))(vx1))
    if vx2 is None:
      if diag:
        return vmap(lambda x: cov_func(x, x))(vx1)
      vx2 = vx1
    return mmap(vx2).T

  return matrix_map


@covariance_matrix
def squared_exponential_kernel(params, x1, x2, warp_func=None):
  """Squared exponential kernel: Eq.(4.9/13) of GPML book.

  Args:
    params: parameters for the kernel.
    x1: a d-diemnsional vector that represent a single datapoint.
    x2: a d-diemnsional vector that represent a single datapoint that can be the
      same as or different from x1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.

  Returns:
    The kernel function evaluation on x1 and x2.
  """
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  r2 = jnp.sum(((x1 - x2) / lengthscale)**2)
  return jnp.squeeze(signal_variance) * jnp.exp(-r2 / 2)


@covariance_matrix
def laplace_kernel(params, x1, x2, warp_func=None):
  """Squared exponential kernel: Eq.(4.9/13) of GPML book.

  Args:
    params: parameters for the kernel.
    x1: a d-diemnsional vector that represent a single datapoint.
    x2: a d-diemnsional vector that represent a single datapoint that can be the
      same as or different from x1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.

  Returns:
    The kernel function evaluation on x1 and x2.
  """
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  r1 = jnp.sum((jnp.abs(x1 - x2) / lengthscale))
  return jnp.squeeze(signal_variance) * jnp.exp(-r1)


@covariance_matrix
def additive_laplace_kernel(params, x1, x2, warp_func=None):
  """Squared exponential kernel: Eq.(4.9/13) of GPML book.

  Args:
    params: parameters for the kernel.
    x1: a d-diemnsional vector that represent a single datapoint.
    x2: a d-diemnsional vector that represent a single datapoint that can be the
      same as or different from x1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.

  Returns:
    The kernel function evaluation on x1 and x2.
  """
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  r1 = jnp.abs(x1 - x2) / lengthscale
  return jnp.squeeze(signal_variance) * jnp.sum(jnp.exp(-r1))


@covariance_matrix
def squared_exponential_sphere_kernel(params, x1, x2, warp_func=None):
  """Squared exponential kernel on sphere distance.

  Args:
    params: parameters for the kernel.
    x1: a d-diemnsional vector that represent a single datapoint.
    x2: a d-diemnsional vector that represent a single datapoint that can be the
      same as or different from x1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.

  Returns:
    The kernel function evaluation on x1 and x2.
  """
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  cosine = get_cosine(params, x1, x2)
  cosine = jnp.clip(cosine, 0.0, 1.0)
  r = jnp.arccos(cosine)
  r2 = (r / lengthscale)**2
  return jnp.squeeze(signal_variance) * jnp.exp(-r2 / 2)


@covariance_matrix
def dot_product_kernel(params, x1, x2, warp_func=None):
  r"""Dot product kernel with normalized inputs.

  Args:
    params: parameters for the kernel. s=params['dot_prod_sigma'] and
      b=params['dot_prod_bias'] corresponds to k(x, x') = b^2 + x^Tx' / s^2,
      where s and b are floats.
    x1: a d-diemnsional vector that represent a single datapoint.
    x2: a d-diemnsional vector that represent a single datapoint that can be the
      same as or different from x1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.

  Returns:
    The kernel function evaluation on x1 and x2.
  """
  params_keys = ['dot_prod_sigma', 'dot_prod_bias']
  sigma, bias = retrieve_params(params, params_keys, warp_func)
  # r1 = jnp.sqrt(jnp.sum(x1**2))
  # r2 = jnp.sqrt(jnp.sum(x2**2))
  # return jnp.dot(x1, x2.T) / jnp.square(sigma) / r1 / r2  + jnp.square(bias)
  return jnp.dot(x1, x2.T) / jnp.square(sigma) + jnp.square(bias)


def get_cosine_phi(params, vx, warp_func=None):
  """Get linear bases for Bayesian linear regression equivalent model."""
  if 'intercept_scaling' in params:
    (intercept_scaling,) = retrieve_params(
        params, ['intercept_scaling'], warp_func
    )
  else:
    intercept_scaling = 1.0
  signal_variance, = retrieve_params(params, ['signal_variance'], warp_func)
  delta = intercept_scaling**2
  r = jnp.sqrt(jnp.sum(vx**2, axis=1, keepdims=True) + delta)
  if intercept_scaling != 0:
    phi = jnp.hstack((vx, jnp.ones((len(vx), 1))*intercept_scaling)) / r
  else:
    phi = vx / r
  return phi * jnp.sqrt(signal_variance)


def get_cosine(params, x1, x2, warp_func=None):
  """Compute cosine between two vectors."""
  # We add a new dimension to x1 and x2 to include the bias term.
  # The results can be very counter intuitive if x range is low.
  # But for image tasks, adding this bias does not seem to matter
  # because activation values are generally very large.
  if 'intercept_scaling' in params:
    (intercept_scaling,) = retrieve_params(
        params, ['intercept_scaling'], warp_func
    )
  else:
    intercept_scaling = 1.0
  delta = intercept_scaling**2
  r1 = jnp.sqrt(jnp.sum(x1**2) + delta)
  r2 = jnp.sqrt(jnp.sum(x2**2) + delta)
  return (jnp.dot(x1, x2.T) + delta) / r1 / r2


@covariance_matrix
def cosine_kernel(params, x1, x2, warp_func=None):
  r"""Kernel defined by cosine similarity.

  Args:
    params: parameters for the kernel.
    x1: a d-diemnsional vector that represent a single datapoint.
    x2: a d-diemnsional vector that represent a single datapoint that can be the
      same as or different from x1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.

  Returns:
    The kernel function evaluation on x1 and x2.
  """
  params_keys = ['signal_variance']
  signal_variance, = retrieve_params(params, params_keys, warp_func)
  return get_cosine(params, x1, x2, warp_func) * signal_variance


def get_latent_var_mu(alpha):
  """Get latent variance and mean for the lognormal distribution in Beta GP."""
  var = jnp.log(1 / alpha + 1)
  mu = jnp.log(alpha) - var / 2
  return var, mu


def set_default_params(params, warp_func=None):
  alpha_eps, = retrieve_params(
      params, ['alpha_eps'], warp_func=warp_func
  )
  params['signal_variance'] = jnp.log(1/alpha_eps + 1)
  y, _ = get_latent_observations(params, jnp.zeros((1, 1)), warp_func=warp_func)
  params['constant'] = jnp.min(y)  # (jnp.max(y) + jnp.min(y)) / 2
  return params


def get_latent_observations(params, y, warp_func=None):
  """Get latent observations of Beta GP.

  Args:
    params: dictionary mapping from parameter keys to values.
    y: labels. Each element of y is either 0 or 1. y: n x 1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.

  Returns:
    y: latent function observation.
    var: latent noise.
  """
  alpha_eps, strength = retrieve_params(
      params, ['alpha_eps', 'strength'], warp_func=warp_func
  )
  alpha = (
      jnp.ones((y.shape[0], 2)) * alpha_eps + jnp.hstack([y, 1 - y]) * strength
  )
  var, y = get_latent_var_mu(alpha)
  return y, var


def gp_predict(
    *,
    mean_func,
    cov_func,
    params,
    x_query,  # n' x d array
    x_observed=None,  # n x d array or None
    y_observed=None,  # n x 1 array or None
    var_observed=None,  # flat array of size n or None
    warp_func=None,
    var_only=True,
    predict_weight=False,
):
  """Predict GP posterior with observed heteroscedastic noise variance.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector.
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix.
    params: dictionary mapping from parameter keys to values.
    x_query: n' x d input array to be queried.
    x_observed: observed n x d input array, or None if no observations.
    y_observed: observed n x 1 evaluations on the input x_observed. Each element
      must be 0 or 1.
    var_observed: observed noise variance vector of shape (n,).
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    var_only: return variance only if True; otherwise return the full covariance
      matrix.
    predict_weight: for cosine kernel only. If True, return the mean and
      covariance of weights in addition to mu and cov.

  Returns:
    mu: posterior mean (n' x 1) if observations are given; otherwise return the
    prior mean (n' x 1).
    cov: posterior or prior variance (n' x 1) if var_only is True; otherwise
    return the posterior or prior covariance matrix (n' x n').
  """
  mu_query = mean_func(params, x_query, warp_func=warp_func)
  cov_query = cov_func(params, x_query, warp_func=warp_func, diag=var_only)
  if (
      x_observed is None
      or x_observed.shape[0] == 0
      or y_observed is None
      or y_observed.shape[0] == 0
      or var_observed is None
      or var_observed.shape[0] == 0
  ):
    return mu_query, cov_query[:, None]
  mu_observed = mean_func(params, x_observed, warp_func=warp_func)
  cov_observed = cov_func(params, x_observed, warp_func=warp_func) + jnp.diag(
      var_observed.flatten()
  )
  chol = jspla.cholesky(cov_observed, lower=True)
  delta_y = y_observed - mu_observed
  kinvy = jspla.cho_solve((chol, True), delta_y)
  cov_observed_query = cov_func(
      params, x_observed, x_query, warp_func=warp_func
  )
  mu = jnp.dot(cov_observed_query.T, kinvy) + mu_query
  v = jspla.solve_triangular(chol, cov_observed_query, lower=True)
  if var_only:
    diagdot = jax.vmap(lambda x: jnp.dot(x, x.T))
    var = cov_query - diagdot(v.T)
    cov = var[:, None]
  else:
    cov = cov_query - jnp.dot(v.T, v)
  if predict_weight:
    phi = get_cosine_phi(params, x_observed, warp_func=warp_func)  # n x d
    u = jnp.dot(phi.T, kinvy)
    sig = jspla.solve_triangular(chol, phi, lower=True)
    sig = jnp.diag(jnp.ones((phi.shape[1],))) - jnp.dot(sig.T, sig)
    return mu, cov, u, sig
  return mu, cov


def beta_gp_predict(
    *,
    mean_func,
    cov_func,
    params,
    x_query,
    x_observed=None,
    y_observed=None,
    warp_func=None,
    var_only=True,
):
  """Predict Beta GP posterior for the latent function.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector.
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix.
    params: dictionary mapping from parameter keys to values.
    x_query: n' x d input array to be queried.
    x_observed: observed n x d input array.
    y_observed: observed n x 1 evaluations on the input x_observed. Each element
      must be 0 or 1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    var_only: return variance only if True; otherwise return the full covariance
      matrix.

  Returns:
    Predictions: a list of tuples. Each tuple is the posterior mean (n' x 1) and
    (co)variance (n' x n' or n' x 1) for the two functions.
  """
  if y_observed is None or y_observed.shape[0] == 0:
    y_latent, var_latent = None, None
  else:
    y_latent, var_latent = get_latent_observations(
        params, y_observed, warp_func=warp_func
    )  # n x 2, n x 2
  # hacky way of setting constant mean
  params = set_default_params(params, warp_func=warp_func)
  predictions = []
  for i in range(2):
    if y_latent is not None:
      y_observed = y_latent[:, i:i+1]
      var_observed = var_latent[:, i]
    else:
      y_observed, var_observed = None, None
    mu_var = gp_predict(
        mean_func=mean_func,
        cov_func=cov_func,
        params=params,
        x_query=x_query,
        x_observed=x_observed,
        y_observed=y_observed,
        var_observed=var_observed,
        warp_func=warp_func,
        var_only=var_only,
    )
    predictions.append(mu_var)
  return predictions


def get_logistic_quantiles(mu, var, q):
  """Get the median and (q, 1-q) quantiles for logistic transform."""
  y = 1 / (1 + jnp.exp(-mu))
  quantiles = jsp.stats.norm.ppf([q, 1 - q], loc=mu, scale=jnp.sqrt(var))  # pytype: disable=wrong-arg-types
  quantiles = 1 / (1 + jnp.exp(-quantiles))
  return y, quantiles


def get_latent_gp(predictions):
  mu, var = (
      predictions[0][0] - predictions[1][0],  # delta of mu
      predictions[0][1] + predictions[1][1],  # sum of var or cov
  )
  return mu, var


def get_beta_quantiles(predictions, q):
  """Get the median and (q, 1-q) quantiles of Beta GP classification model."""
  mu, var = get_latent_gp(predictions)
  return get_logistic_quantiles(mu, var, q)


def get_mvn_samples(mu, cov, key=None, n=int(1e6)):
  if mu.shape[0] != cov.shape[0] or cov.shape[0] != cov.shape[1]:
    raise ValueError(
        f'mu.shape={mu.shape} and cov.shape={cov.shape}. Mean and cov shape'
        ' must match. Cov must be a square matrix.'
    )
  if key is None:
    key = jax.random.PRNGKey(0)
  norm_samples = jax.random.normal(key, (mu.shape[0], n))
  chol = jspla.cholesky(cov, lower=True)
  samples = jnp.dot(chol, norm_samples) + mu
  return samples, chol


def beta_gp_uncertainty(predictions, seed=0, n=int(1e6)):
  """Get measures of aleatory and epistemic uncertainties for a batch of queries.

  Args:
    predictions: Predictions returned by beta_gp_nll with var_only=True.
    seed: int random seed.
    n: number of samples for Monte Carlo estimation.

  Returns:
    Dictionary mapping from name to measure.
  """
  latent_mu, latent_var = get_latent_gp(predictions)
  if latent_var.shape[0] == latent_var.shape[1]:
    # predictions include covaraince instead of variance.
    latent_var = jnp.diag(latent_var)[:, None]
  latent_var = jnp.maximum(latent_var, 1e-32)
  return gp_uncertainty(latent_mu, latent_var, seed=seed, n=n)


def gp_uncertainty(latent_mu, latent_var, seed=0, n=int(1e6)):
  """Measure uncertainty metrics for logistic GP."""
  key = jax.random.PRNGKey(seed)
  norm_samples = jax.random.normal(key, (latent_mu.shape[0], n))
  iid_samples = norm_samples * jnp.sqrt(latent_var) + latent_mu
  p_samples = 1.0 / (1 + jnp.exp(-iid_samples))  # num_inputs x n
  ret = classifier_samples_uncertainty(p_samples)
  norm_entropy = 0.5 * jnp.log(2 * jnp.pi * latent_var) + 0.5
  entropy = (
      norm_entropy
      + latent_mu
      - 2 * jnp.mean(jnp.log(1 + jnp.exp(iid_samples)), axis=1, keepdims=True)
  )
  ret.update({
      'epistemic_entropy': entropy,  # Entropy of each label distribution
      'latent_var': latent_var,
      'latent_mu': latent_mu,
      'Episteme': -entropy,
  })
  return ret


def classifier_samples_uncertainty(
    p_samples,  # num_inputs x n
    approx_entropy=False,  # approximate entropy with Gaussian entropy if True
):
  """Measure uncertainty metrics with classifier samples."""
  mu = jnp.mean(p_samples, axis=1, keepdims=True)  # num_inputs x 1
  var = jnp.mean(p_samples**2, axis=1, keepdims=True) - (
      mu**2
  )  # num_inputs x 1
  bernoulli_var = jnp.mean(p_samples * (1 - p_samples), axis=1, keepdims=True)
  bernoulli_entropy = -jnp.mean(
      p_samples * jnp.log(p_samples) + (1 - p_samples) * jnp.log(1 - p_samples),
      axis=1,
      keepdims=True,
  )
  info_gain = (
      -(mu * jnp.log(mu) + (1 - mu) * jnp.log(1 - mu)) - bernoulli_entropy
  )
  ret = {
      'epistemic_var': var,  # Variance for each label
      'bernoulli_mu': mu,  # Mean prediction of label, i.e., Bernoulli param
      'expected_aleatory_entropy': (
          bernoulli_entropy
      ),  # Expected Bernoulli entropy
      'expected_aleatory_var': bernoulli_var,  # Expected Bernoulli variance
      'information_gain': info_gain,  # I(label; function | input, data)
      'Alea': bernoulli_entropy,
      'Judged probability': mu,
  }
  if approx_entropy:
    # use Gaussian entropy to approximate the epistemic entropy
    var = jnp.maximum(var, 1e-32)
    entropy = 0.5 * jnp.log(2 * jnp.pi * var) + 0.5
    ret['epistemic_entropy'] = entropy
    ret['Episteme'] = -entropy
  return ret


def compute_entropy_reduction(
    *,
    mean_func,
    cov_func,
    params,
    x_of_interest,
    x_to_observe,
    x_observed=None,
    y_observed=None,
    warp_func=None,
    seed=0,
    n=int(1e6),
    n_e=10,
):
  """Expected entropy reduction of labels for x_of_interest by observing x_to_observe.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector.
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix.
    params: dictionary mapping from parameter keys to values.
    x_of_interest: n' x d input array to be queried.
    x_to_observe: n'' x d input array. Each input in x_to_observe is assumed to
      be observed, and we measure how much entropy can be reduced by observing
      this input.
    x_observed: observed n x d input array.
    y_observed: observed n x 1 evaluations on the input x_observed. Each element
      must be 0 or 1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    seed: int random seed.
    n: number of samples for Monte Carlo estimation of entropy.
    n_e: number of samples for Monte Carlo estimation of the expected entropy
      over random labels of x_to_observe.

  Returns:
    Entropy reduction for each .
  """
  predictions = beta_gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      params=params,
      x_query=x_of_interest,
      x_observed=x_observed,
      y_observed=y_observed,
      warp_func=warp_func,
      var_only=False,
  )
  mu_query, cov_query = get_latent_gp(predictions)
  key = jax.random.PRNGKey(seed)
  key, subkey = jax.random.split(key)
  entropy_before = estimate_batch_entropy(mu_query, cov_query, key=subkey, n=n)
  predictions = beta_gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      params=params,
      x_query=x_to_observe,
      x_observed=x_observed,
      y_observed=y_observed,
      warp_func=warp_func,
      var_only=True,
  )
  mu_to_observe, var_to_observe = get_latent_gp(predictions)

  def vmap_func(index_key):
    x_index, key = index_key
    x_new_observed = jnp.vstack([x_observed, x_to_observe[x_index]])
    mu, var = mu_to_observe[x_index], var_to_observe[x_index]
    norm_samples = jax.random.normal(key, (n_e,)) * jnp.sqrt(var) + mu

    def inner_vmap_func(y):
      # TODO(wangzi): use rank-1 update to speed up prediction.
      y_new_observed = jnp.vstack([y_observed, y])
      predictions = beta_gp_predict(
          mean_func=mean_func,
          cov_func=cov_func,
          params=params,
          x_query=x_of_interest,
          x_observed=x_new_observed,
          y_observed=y_new_observed,
          warp_func=warp_func,
          var_only=False,
      )
      new_mu, new_cov = get_latent_gp(predictions)
      return estimate_batch_entropy(new_mu, new_cov, key=key, n=n)

    return jnp.mean(inner_vmap_func(norm_samples))

  vmap_args = [
      [i, k[0]]
      for i, k in enumerate(jax.random.split(key, x_to_observe.shape[0]))
  ]
  vmap_args = jnp.array(vmap_args)
  entropy_after = vmap(vmap_func)(vmap_args)
  return entropy_before - entropy_after


def estimate_entropy_from_mu_chol(mu, chol, samples):
  """Monte Carlo estimation of entropy for Beta GP using n samples."""
  if (
      mu.shape[0] != chol.shape[0]
      or chol.shape[0] != chol.shape[1]
      or samples.shape[0] != mu.shape[0]
  ):
    raise ValueError(
        f'mu.shape={mu.shape}, chol.shape={chol.shape},'
        f' samples.shape={samples.shape}. Mean and chol shape must match. chol'
        ' must be a square matrix.'
    )

  mvn_entropy = jnp.sum(jnp.log(jnp.diag(chol))) + 0.5 * mu.shape[0] * (
      jnp.log(2 * jnp.pi) + 1.0
  )
  sum_mu = jnp.sum(mu)
  additional_term = 2 * jnp.sum(jax.nn.softplus(samples)) / samples.shape[1]
  return mvn_entropy + sum_mu - additional_term


def estimate_batch_entropy(mu, cov, key=None, n=int(1e6)):
  """Monte Carlo estimation of entropy for Beta GP using n samples."""
  if key is None:
    key = jax.random.PRNGKey(0)
  samples, chol = get_mvn_samples(mu, cov, key=key, n=n)
  return estimate_entropy_from_mu_chol(mu, chol, samples)


def mvn_nll(y, mu, cov):
  """Negative log likelihood for one sample of Multivariate Normal."""
  if y.shape != mu.shape:
    raise ValueError('Shape of y and mu must match.')
  if y.shape[1] != 1 or mu.shape[1] != 1:
    raise ValueError('y and mu must be column vectors.')
  if y.shape[0] != cov.shape[0]:
    raise ValueError('Shape of y and cov must match.')
  if cov.shape[0] != cov.shape[1]:
    raise ValueError('Cov must be a square matrix.')
  y = y - mu
  chol = jspla.cholesky(cov, lower=True)
  kinvy = jspla.cho_solve((chol, True), y)
  return jnp.sum(
      0.5 * jnp.dot(y.T, kinvy)
      + jnp.sum(jnp.log(jnp.diag(chol)))
      + 0.5 * y.shape[0] * jnp.log(2 * jnp.pi)
  )


def beta_mnll(
    mean_func,
    cov_func,
    params,
    x_query,
    y_query,
    x_train,  # n x d
    y_train,  # n x num_classes
    warp_func=None,
):
  """MNLL Eq 2 of Milios et al., 2018."""
  if len(y_train.shape) == 1:
    y_train = y_train[:, None]
  predictions = beta_gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      params=params,
      x_query=x_query,
      x_observed=x_train,
      y_observed=y_train,
      warp_func=warp_func,
      var_only=True,
  )
  alpha = 1 / (jnp.exp(predictions[0][1]) - 1)
  beta = 1 / (jnp.exp(predictions[1][1]) - 1)
  mnll = jnp.where(y_query, alpha / (alpha + beta), beta / (alpha + beta))
  return -jnp.sum(jnp.log(mnll))


def beta_gp_nll(
    mean_func,
    cov_func,
    params,
    x_train,  # n x d
    y_train,  # n x num_classes
    warp_func=None,
):
  """Negative log data likelihood for Beta GP."""
  if len(y_train.shape) == 1:
    y_train = y_train[:, None]
  # hacky way of setting constant mean
  params['constant'] = set_default_params(params, warp_func=warp_func)
  predictions = beta_gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      params=params,
      x_query=x_train,
      warp_func=warp_func,
      var_only=False,
  )

  def vmap_func(y):
    if len(y.shape) == 1:
      y = y[:, None]
    y_latent, var_latent = get_latent_observations(
        params, y, warp_func=warp_func
    )
    nll = []
    for i in range(y_latent.shape[1]):
      var = var_latent[:, i]
      cov = predictions[i][1] + jnp.diag(var.flatten())
      nll.append(mvn_nll(y_latent[:, i:i+1], predictions[i][0], cov))
    return jnp.sum(jnp.array(nll))

  return vmap(vmap_func)(y_train.T).mean(axis=0)


def get_probit_quantiles(mu, var, q):
  """Get the median and (q, 1-q) quantiles for cumulative Gaussian transform."""
  y = jsp.stats.norm.cdf(mu.flatten())
  quantiles = jsp.stats.norm.cdf(
      jsp.stats.norm.ppf(  # pytype: disable=wrong-arg-types
          [q, 1 - q], loc=mu / jnp.sqrt(var + 1), scale=jnp.sqrt(var)
      )
  )
  return y, quantiles


def gpr_predict(
    mean_func,
    cov_func,
    params,
    x_query,
    x_observed=None,
    y_observed=None,
    warp_func=None,
    var_only=True,
):
  """Predict GP (classification as regression) posterior for the latent function.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector.
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix.
    params: dictionary mapping from parameter keys to values.
    x_query: n' x d input array to be queried.
    x_observed: observed n x d input array.
    y_observed: observed n x 1 evaluations on the input x_observed. Each element
      must be 0 or 1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    var_only: return variance only if True; otherwise return the full covariance
      matrix.

  Returns:
    Posterior mean (n' x 1) and variance (n' x 1) for query points x_query.
  """
  noise_variance, scale = retrieve_params(
      params, ['noise_variance', 'scale'], warp_func=warp_func
  )
  y_observed = (y_observed * 2 - 1) * scale  # rescaled to [-scale, scale]
  var_observed = jnp.ones(x_observed.shape[0]) * noise_variance
  return gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      params=params,
      x_query=x_query,
      x_observed=x_observed,
      y_observed=y_observed,
      var_observed=var_observed,
      warp_func=warp_func,
      var_only=var_only,
  )
