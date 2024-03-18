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

"""Tests for wordnet."""

from absl.testing import absltest
from absl.testing import parameterized
from gpax.probing import wordnet
import networkx as nx


class WordnetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.dag = wordnet.get_dag([
        'cat.n.01',
        'dog.n.01',
        'deer.n.01',
        'crow.n.01',
        'owl.n.01',
        'snake.n.01',
        'lizard.n.01',
    ])

  def test_structure(self):
    self.assertSameStructure(
        nx.node_link_data(self.dag),
        {
            'directed': True,
            'graph': {},
            'links': [
                {'source': 'carnivore.n.01', 'target': 'cat.n.01'},
                {'source': 'carnivore.n.01', 'target': 'dog.n.01'},
                {'source': 'placental.n.01', 'target': 'carnivore.n.01'},
                {'source': 'placental.n.01', 'target': 'deer.n.01'},
                {'source': 'vertebrate.n.01', 'target': 'bird.n.01'},
                {'source': 'vertebrate.n.01', 'target': 'placental.n.01'},
                {'source': 'vertebrate.n.01', 'target': 'diapsid.n.01'},
                {'source': 'animal.n.01', 'target': 'vertebrate.n.01'},
                {'source': 'animal.n.01', 'target': 'dog.n.01'},
                {'source': 'bird.n.01', 'target': 'crow.n.01'},
                {'source': 'bird.n.01', 'target': 'owl.n.01'},
                {'source': 'diapsid.n.01', 'target': 'snake.n.01'},
                {'source': 'diapsid.n.01', 'target': 'lizard.n.01'},
            ],
            'multigraph': False,
            'nodes': [
                {'id': 'cat.n.01'},
                {'id': 'carnivore.n.01'},
                {'id': 'placental.n.01'},
                {'id': 'vertebrate.n.01'},
                {'id': 'animal.n.01'},
                {'id': 'dog.n.01'},
                {'id': 'deer.n.01'},
                {'id': 'crow.n.01'},
                {'id': 'bird.n.01'},
                {'id': 'owl.n.01'},
                {'id': 'diapsid.n.01'},
                {'id': 'snake.n.01'},
                {'id': 'lizard.n.01'},
            ],
        },
    )

  @parameterized.parameters(
      (None, 'animal.n.01'),
      (['cat.n.01'], 'cat.n.01'),
      (['cat.n.01', 'dog.n.01'], 'carnivore.n.01'),
      (['cat.n.01', 'deer.n.01'], 'placental.n.01'),
      (['cat.n.01', 'crow.n.01'], 'vertebrate.n.01'),
      (['crow.n.01', 'owl.n.01'], 'bird.n.01'),
      (['snake.n.01', 'lizard.n.01'], 'diapsid.n.01'),
  )
  def test_get_lca(self, nodes, lca):
    self.assertEqual(wordnet.get_lca(self.dag, nodes), lca)


if __name__ == '__main__':
  absltest.main()
