from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

import where


def default_settings(**params):
  return where.default_settings(parent_factor=1.25,
                                max_factor=0.75).update(**params)


def build(dataset, rows=None, **params):
  if rows is None:
    rows = dataset.get_rows()
  settings = default_settings(**params)
  root_node = where.build(dataset, rows, **settings.has())
  pruned_rows = []
  leaves = [tree_leaf for tree_leaf, _ in where.get_leaves(root_node)]
  max_var = where.max_tree_variance(root_node)
  a, b = settings.parent_factor, settings.max_factor
  for leaf in leaves:
    if (leaf.variance < a * leaf.parent.variance) and (leaf.variance < b * max_var):
      pruned_rows += leaf.get_rows()
  return where.build(dataset, pruned_rows, **settings.has())


def teak(dataset, test, train):
  node = build(dataset, train)
  predicts = []
  for test_row in test:
    test_leaf = where.get_leaf(dataset, test_row, node)
    nearest_row = where.closest(dataset, test_row, test_leaf.get_rows())
    predicts.append(dataset.effort(nearest_row))
  return predicts
