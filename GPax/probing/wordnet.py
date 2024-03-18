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

"""Construct the WordNet hierarchy with NLTK and NetworkX.

"""

import networkx as nx
from nltk.corpus import wordnet as wn  # pylint: disable=g-importing-member


def get_dag(words, prune=True):
  """Constructs a dag that represents the WordNet hierarchy.

  Args:
    words: The sinks of the dag (https://www.nltk.org/howto/wordnet.html).
    prune: Prune half-nodes from the dag.

  Returns:
    A directed acyclic graph.
  """
  dag = nx.DiGraph()

  for word in words:
    queue = [wn.synset(word)]
    while queue:
      synset = queue.pop()
      for hypernym in synset.hypernyms():
        queue.append(hypernym)
        dag.add_edge(hypernym.name(), synset.name())

  while prune and (nodes := [n for n, d in dag.out_degree() if d == 1]):
    node = nodes[0]  # Half-node
    child = next(dag.neighbors(node))
    for parent in dag.predecessors(node):
      dag.add_edge(parent, child)
    dag.remove_node(node)

  assert nx.is_directed_acyclic_graph(dag)
  return nx.freeze(dag)


def get_sinks(dag, source=None):
  """Returns the sinks of a dag."""
  nodes = None if source is None else nx.dfs_preorder_nodes(dag, source)
  return [n for n, d in dag.out_degree(nodes) if d == 0]


def get_lca(dag, nodes=None):
  """Returns the least common ancestor a set of nodes."""
  if nodes is None:
    return next(nx.topological_sort(dag))
  root = nodes[0]
  for node in nodes[1:]:
    root = nx.lowest_common_ancestor(dag, root, node)
  return root


def get_subgraph(dag, nodes):
  """Constructs the subgraph that contains a set of nodes."""
  lca = get_lca(dag, nodes)
  sub = nx.subgraph(dag, nx.dfs_preorder_nodes(dag, lca))
  return nx.freeze(sub)
