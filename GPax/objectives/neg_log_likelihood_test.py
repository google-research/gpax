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

"""Test for neg_log_likelihood.py."""

import logging

from absl.testing import absltest
from gpax import utils
from gpax.models import gp
from gpax.objectives import neg_log_likelihood as nll
from jax import numpy as jnp
from jax import random


class NLLTest(absltest.TestCase):

  def test_objective(self):
    key1 = random.PRNGKey(0)
    dataset = [
        utils.SubDataset(
            random.uniform(key1, (8, 5)), random.uniform(key1, (8,))),
        utils.SubDataset(
            random.uniform(key1, (8, 5)), random.uniform(key1, (8,)))
    ]

    model = gp.GaussianProcess()
    params = model.init(key1, dataset[0].x)
    objective = nll.objective(model, params, dataset)

    self.assertNotEqual(objective, jnp.nan)
    self.assertEmpty(objective.shape)
    logging.info(msg=f'nll = {objective}')


if __name__ == '__main__':
  absltest.main()
