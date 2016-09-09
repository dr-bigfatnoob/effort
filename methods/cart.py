from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from sklearn.tree import DecisionTreeRegressor


def cart_format(dataset, test, train):
  def format_row(line):
    formatted = []
    decisions = line.cells[:len(dataset.dec_meta)]
    for i, (val, meta) in enumerate(zip(decisions, dataset.dec_meta)):
      if not meta.has_attr("ignores") or i not in meta.ignores:
        formatted.append(val)
    return formatted
  train_input_set, train_output_set = [], []
  test_input_set, test_output_set = [], []
  for row in train:
    train_input_set += [format_row(row)]
    train_output_set += [dataset.effort(row)]
  for row in test:
    test_input_set += [format_row(row)]
    test_output_set += [dataset.effort(row)]
  return train_input_set, train_output_set, test_input_set, test_output_set


def cart(dataset, test, train):
  train_ip, train_op, test_ip, test_op = cart_format(dataset, test, train)
  dec_tree = DecisionTreeRegressor(criterion="mse", random_state=1)
  dec_tree.fit(train_ip, train_op)
  return dec_tree.predict(test_ip)
