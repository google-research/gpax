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

"""Tests for acquisitions."""

from absl.testing import absltest
from absl.testing import parameterized
from gpax.bayesopt import acquisitions
import jax  # pylint:disable=unused-import
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


class AcquisitionsTest(parameterized.TestCase):

  def test_quantile(self):
    acq = acquisitions.Quantile(0.5)
    self.assertAlmostEqual(acq.evaluate(tfd.Normal(0, 1)), 0.)

  def test_ucb(self):
    acq = acquisitions.UpperConfidenceBound(2.)
    self.assertAlmostEqual(acq.evaluate(tfd.Normal(0.1, 1)), 2.1)

  @parameterized.parameters((acquisitions.Goal.MAXIMIZE, 0.10043107),
                            (acquisitions.Goal.MINIMIZE, 1.0004311))
  def test_ei(self, goal, expected):
    acq = acquisitions.ExpectedImprovement(1., goal)
    self.assertAlmostEqual(acq.evaluate(tfd.Normal(0.1, 1)), expected, places=5)

  @parameterized.parameters((acquisitions.Goal.MAXIMIZE, .9),
                            (acquisitions.Goal.MINIMIZE, -.9))
  def test_zscore(self, goal, expected):
    acq = acquisitions.ImprovementZScore(1., goal)
    self.assertAlmostEqual(acq.evaluate(tfd.Normal(0.1, 1)), expected)

  def test_ts(self):
    acq = acquisitions.ThomsonSampling(jax.random.PRNGKey(1))
    self.assertSequenceEqual(acq.evaluate(tfd.Normal(0.1, 1)).shape, [])


if __name__ == '__main__':
  absltest.main()
