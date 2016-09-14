from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import *


class Operator(O):
  """
  Class indicating an operator(+, -, *)
  """
  def __init__(self, sym):
    O.__init__(sym)
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


class COGEE(O):
  def __init__(self, dataset, rows):
    O.__init__(self)
    self.dataset = dataset
    self.rows = rows



