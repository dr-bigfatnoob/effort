from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import EPS
import where


def peeking2(dataset, test, train):
  node = where.build(dataset, train)
  predicts = []
  for test_row in test:
    test_leaf = where.get_leaf(dataset, test_row, node)
    nearest_rows = where.closest_n(dataset, test_row, test_leaf.get_rows(), 2)
    wt_0 = nearest_rows[1][0] / (nearest_rows[0][0] + nearest_rows[1][0] + EPS)
    wt_1 = nearest_rows[0][0] / (nearest_rows[0][0] + nearest_rows[1][0] + EPS)
    predicts.append(dataset.effort(nearest_rows[0][1]) * wt_0 + dataset.effort(nearest_rows[1][1]) * wt_1)
  return predicts
