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

"""Common utils for GPax."""
from typing import NamedTuple, Optional, Tuple, Union, Any, Sequence

from flax.core import frozen_dict
import jax
from jax import tree_util
import jax.numpy as jnp
import numpy as np

Array = Any


class SubDataset(NamedTuple):
  """Sub dataset with x: n x d and y: n x m; d, m>=1."""
  x: jnp.ndarray
  y: jnp.ndarray
  aligned: Optional[Union[int, str, Tuple[str, ...]]] = None

Dataset = Sequence[SubDataset]


class ParamsTree:
  """Converts between FrozenDict and flat array.

  Flax modules' init() returns a FrozenDict that looks like:

  FrozenDict({
    params: {
        Dense_0: {
            kernel: DeviceArray([[ 0.9998795 , -0.1408551 , -0.9979557 ],
                         [ 1.4667332 ,  0.59636754,  0.38262498]],
                         dtype=float32),
            bias: DeviceArray([0., 0., 0.], dtype=float32),
        },
    },
  })

  (Note: this example was created from a 2x3 Dense layer) If params is the
  above FrozenDict, then:

  tree = Tree(params)
  arr = tree.toarray(params)
  # arr is an array of shape (9,) with values equal to
  # [ 0.          0.          0.          0.9998795  -0.1408551  -0.9979557
  # 1.4667332   0.59636754  0.38262498]
  dic = tree.todict(arr)
  # dic has the same values as params.
  """

  def __init__(self, params: frozen_dict.FrozenDict):
    params, self.tree = tree_util.tree_flatten(params)
    self.shapes = tree_util.tree_map(lambda x: x.shape, params)
    sizes = np.array([np.prod(s) for s in self.shapes])
    self.indices = np.cumsum(sizes)[:-1]

  def toarray(self, params: frozen_dict.FrozenDict) -> Array:
    params = tree_util.tree_flatten(params)[0]
    return jnp.concatenate([jnp.reshape(p, [-1]) for p in params])

  def todict(self, params: Array) -> frozen_dict.FrozenDict:
    params = jnp.split(params, self.indices)
    params = [jnp.reshape(p, self.shapes[i]) for i, p in enumerate(params)]
    params = tree_util.tree_unflatten(self.tree, params)
    return params

  def __repr__(self) -> str:
    return f'shapes={self.shapes}, indices={self.indices}'


def constant_initializer_factory(constant: float):

  def initializer(*args, **kwargs):
    return jax.nn.initializers.ones(*args, **kwargs) * constant

  return initializer
