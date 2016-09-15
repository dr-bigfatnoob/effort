from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import *
from math import sqrt
import numpy as np
import scipy.stats as stats


def rand():
  return random.uniform(0, 1e6)


def absolute_errors(actuals, estimated):
  return [abs(a - e) for a, e in zip(actuals, estimated)]


def confidence(x, n, k, p):
  """
  Compute Confidence based on Student's t-quantiles
  :param x: array
  :param n: number of samples
  :param k: number of parameters
  :param p: confidence
  :return: value from array
  """
  x_s = sorted(x)
  df = n - k
  t_dist = stats.t.cdf(x_s, df)
  phi = None
  for x_i, t_i in zip(x_s, t_dist):
    if t_i > p:
      phi = x_i
      break
  if phi is None:
    phi = x_s[-1]
  return phi * np.std(x) / sqrt(n)


class Operator(O):
  """
  Class indicating an operator(+, -, *)
  """
  def __init__(self, sym):
    O.__init__(self)
    self.sym = sym

  def op(self, a, b):
    assert False

  def __hash__(self):
    return hash(self.sym)

  def __eq__(self, other):
    return self.sym == other.sym


class Add(Operator):
  def __init__(self):
    Operator.__init__(self, "+")

  def op(self, a, b):
    return a + b


class Subtract(Operator):
  def __init__(self):
    Operator.__init__(self, "-")

  def op(self, a, b):
    return a - b


class Multiply(Operator):
  def __init__(self):
    Operator.__init__(self, "*")

  def op(self, a, b):
    return a * b


class Constant(O):
  def __init__(self, value):
    O.__init__(self)
    self.value = value

  def __hash__(self):
    return self.value

  def __eq__(self, other):
    return self.value == other.value


class Variable(O):
  def __init__(self, name):
    O.__init__(self, name=name)


class Point(O):
  def __init__(self, sequence, dataset, rows):
    O.__init__(self)
    self.sequence = sequence
    self.objectives = None
    self._dataset = dataset
    self._rows = rows

  def evaluate_row(self, row):
    string = ""
    for i, meta in enumerate(self._dataset.dec_meta):
      string += str(self.sequence[4 * i].value)
      string += str(self.sequence[4 * i + 1].sym)
      string += str(row.cells[i])
      string += str(self.sequence[4 * i + 3].sym)
    string += str(self.sequence[-1].value)
    return eval(string)

  def evaluate_rows(self):
    return [self.evaluate_row(row) for row in self._rows]

  def compute_objectives(self):
    if self.objectives is None:
      actuals = [self._dataset.effort(row) for row in self._rows]
      computed = self.evaluate_rows()
      errors = absolute_errors(actuals, computed)
      sae = sum(errors)
      conf = confidence(errors, len(self._rows), len(self._dataset.dec_meta)+1, 0.95)
      self.objectives = [sae, conf]
    return self.objectives


class COGEE(O):
  operators = [Add(), Subtract(), Multiply()]

  def __init__(self, dataset, rows):
    O.__init__(self)
    self.dataset = dataset
    self.rows = rows
    self.sequence = None

  def generate_one(self):
    dataset = self.dataset
    sequence = []
    for meta in dataset.dec_meta:
      sequence.append(Constant(rand()))
      sequence.append(random.choice(COGEE.operators))
      sequence.append(Variable(meta.name))
      sequence.append(random.choice(COGEE.operators))
    sequence.append(Constant(rand()))
    return Point(sequence, self.dataset, self.rows)

  def populate(self, pop_size):
    points = []
    for _ in range(pop_size):
      points.append(self.generate_one())
    return points

  def run(self, pop_size=100, gens=100):
    gen = 0
    population = self.populate(pop_size)
    while gen < gens:
      gen += 1
    print(population[0].compute_objectives())
