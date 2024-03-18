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

"""Probing utilities."""

import jax
from jax import tree_util
import jax.numpy as jnp
import tensorflow as tf


def pad_along_axis(x, before=0, after=0, axis=0, **kwargs):
  """Pads a tensor along a given axis."""
  pad_width = [(0, 0)] * x.ndim
  pad_width[axis] = (before, after)
  return jnp.pad(x, pad_width, **kwargs)


def reshard(x):
  """Shards a tensor before a pmap call."""
  assert x.shape[0] % jax.device_count() == 0
  return x.reshape([jax.device_count(), -1, *x.shape[1:]])


def unshard(x):
  """Unshards a tensor after a pmap call."""
  assert x.shape[0] == jax.device_count()
  return x.reshape([-1, *x.shape[2:]])


def _padded_pmap_call(pmap_func, tree, padded_batch_size=None):
  """Calls a pmap'ed function with a constant batch size."""
  batch_size = tree_util.tree_leaves(tree)[0].shape[0]
  padded_batch_size = padded_batch_size or jax.device_count()
  assert padded_batch_size % jax.device_count() == 0
  assert padded_batch_size >= batch_size
  pad_after = padded_batch_size - batch_size
  tree = tree_util.tree_map(lambda x: pad_along_axis(x, after=pad_after), tree)
  tree = tree_util.tree_map(reshard, tree)
  tree = pmap_func(tree)
  tree = tree_util.tree_map(unshard, tree)
  tree = tree_util.tree_map(lambda x: x[:batch_size], tree)
  return tree


def padded_pmap_loop(pmap_func, tree, padded_batch_size=None):
  """Repeatededly calls a pmap'ed function with a constant batch size.

  Args:
    pmap_func: A pmap'ed function `pmap_func(input_tree) -> output_tree`.
    tree: A pytree of tensors. All tensors must have the same batch size.
    padded_batch_size: A constant batch size to avoid XLA recompilation.
      Defaults to the number of devices. `padded_batch_size` must be less than
      or equal to the batch size of `tree` and divisible by the number of
      devices.

  Returns:
    A pytree of tensors.
  """
  padded_batch_size = padded_batch_size or jax.device_count()
  ds = tf.data.Dataset.from_tensor_slices(tree)
  ds = ds.batch(padded_batch_size)
  batches = [
      _padded_pmap_call(pmap_func, batch, padded_batch_size)
      for batch in ds.as_numpy_iterator()
  ]
  treedef = tree_util.tree_structure(batches[0])
  # TODO(alexku): Convert to numpy to save HBM?
  xs = list(map(jnp.concatenate, zip(*map(tree_util.tree_leaves, batches))))
  return tree_util.tree_unflatten(treedef, xs)
