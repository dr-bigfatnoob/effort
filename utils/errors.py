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


def sa(*args):
  """
  Shepperd and MacDonell's standardized error.
  SA = 1 - MAR/MARp0
  :param args: [actual, predicted, vector]
  :return: SA
  """
  mar = abs(args[0] - args[1])
  mar_p0 = sum([abs(choice(args[2]) - args[0]) for _ in range(1000)]) / 1000
  return mar / mar_p0


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
  Mean Standard Accuracy Error
  :param args: [[actual vals], [predicted vals], [all effort]]
  :return:
  """
  errors = []
  for actual, predicted in zip(args[0], args[1]):
    errors.append(sa(actual, predicted, args[2]))
  return np.mean(errors)


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
