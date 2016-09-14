from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from random import choice
import numpy as np


def rae(*args):
  """
  :param args: [actual, predicted]
  :return: MRE
  """
  return abs(args[0] - args[1]) / args[0]


def rel_error(*args):
  """
  :param args: [actual, predicted]
  :return: Relative Error
  """
  return args[1] - args[0]


def mre(*args):
  """
  Mean Relative Error
  :param args: [[actual vals], [predicted vals], [all effort]]
  :return: mre
  """
  errors = []
  for actual, predicted in zip(args[0], args[1]):
    errors.append(rae(actual, predicted))
  return np.mean(errors)


def msa(*args):
  """
  Mean Standard Accuracy
  :param args: [[actual vals], [predicted vals], [all effort]]
  :return:
  """
  assert len(args[0]) == len(args[1])
  mae = sum([abs(actual - predicted) for actual, predicted in zip(args[0], args[1])]) / len(args[0])
  mae_guess = sum([abs(choice(args[0]) - choice(args[2])) for _ in range(1000)]) / 1000
  return 1 - mae / mae_guess


def msae(*args):
  """
  Mean Standardized Accuracy Error
  :param args:
  :return:
  """
  return 1 - msa(*args)


def re_star(*args):
  """
  Ratio of Variance
  :param args: [[actual vals], [predicted vals], [all effort]]
  :return:
  """
  residuals = []
  for actual, predicted in zip(args[0], args[1]):
    residuals.append(rel_error(actual, predicted))
  if len(residuals) > 1:
    return np.var(residuals) / np.var(args[1])
  else:
    return residuals[0] / args[1][0]
