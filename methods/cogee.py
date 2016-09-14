from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import *


def rand():
  return random.uniform(0, 1e6)


class Operator(O):
  """
  Class indicating an operator(+, -, *)
  """
  def __init__(self, sym):
    O.__init__(self, sym=sym)

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
  def __init__(self, sequence):
    O.__init__(self)
    self.sequence = sequence

  def evaluate_row(self, problem, row):
    string = ""
    for i, meta in enumerate(problem.dec_meta):
      string += str(self.sequence[4 * i].value)
      string += str(self.sequence[4 * i + 1].sym)
      string += str(row.cells[i])
      string += str(self.sequence[4 * i + 3].sym)
    string += str(self.sequence[-1].value)
    return eval(string)

  def evaluate_rows(self, problem, rows):
    return [self.evaluate_row(problem, row) for row in rows]


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
    return Point(sequence)





