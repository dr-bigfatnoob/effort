from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import *
from utils.errors import absolute_errors, confidence
from optimizer.nsga2 import nsga2, NSGAPoint, make_roulette_map, roulette_wheel


def rand():
  return random.uniform(-1e2, 1e2)


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

  def mutate(self):
    lst = ['+', '-', '*']
    lst.remove(self.sym)
    sel = random.choice(lst)
    if sel == '+': return Add()
    elif sel == '-': return Subtract()
    elif sel == '*': return Multiply()
    else: raise RuntimeError("Invalid symbol : %s" % sel)


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

  def mutate(self):
    r = rand()
    while r == self.value:
      r = rand()
    return Constant(rand())


class Variable(O):
  def __init__(self, name):
    O.__init__(self, name=name)

  def mutate(self):
    return self


class CogeePoint(NSGAPoint):
  def __init__(self, sequence, dataset, rows, better=(lt, lt)):
    NSGAPoint.__init__(self, sequence)
    self.sequence = self.decisions
    self._dataset = dataset
    self._rows = rows
    self.better = better

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

  def clone(self):
    new = CogeePoint(self.sequence, self._dataset, self._rows)
    new.objectives = self.objectives
    return new

  def dominates(self, another, problem):
    assert self.objectives is not None
    assert another.objectives is not None
    better = False
    for i, (one, two) in enumerate(zip(self.objectives, another.objectives)):
      if self.better[i](one, two):
        better = True
      elif one != two:
        return False
    return better


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
    return CogeePoint(sequence, self.dataset, self.rows)

  def populate(self, pop_size):
    points = []
    for _ in range(pop_size):
      points.append(self.generate_one())
    return points

  def crossover_mutate(self, mom, dad, cr=0.5, mr=0.1):
    """
    Perform crossover and mutation
    :param mom: Parent 1
    :param dad: Parent 2
    :param cr: Crossover rate
    :param mr: Mutation rate
    :return: [bro, sis]
    """
    r = random.random()
    if r <= cr:
      l = len(mom.sequence)
      split = random.randint(0, l - 1)
      bro = CogeePoint(mom.sequence[:split] + dad.sequence[split:], self.dataset, self.rows)
      sis = CogeePoint(dad.sequence[:split] + mom.sequence[split:], self.dataset, self.rows)
      for i in xrange(l):
        r = random.random()
        if r < mr:
          bro.sequence[i] = bro.sequence[i].mutate()
        r = random.random()
        if r < mr:
          sis.sequence[i] = sis.sequence[i].mutate()
      return [bro, sis]
    else:
      return None

  def run(self, pop_size=100, gens=250, retain_size=10):
    gen = 0
    population = self.populate(pop_size)
    [point.compute_objectives() for point in population]
    while gen < gens:
      population = nsga2(None, population)
      roulette_map = make_roulette_map(population)
      children = []
      for _ in xrange(len(population)):
        [mom, dad] = roulette_wheel(roulette_map, 2)
        kids = self.crossover_mutate(mom, dad)
        if kids is None:
          continue
        kids[0].compute_objectives()
        kids[1].compute_objectives()
        children += kids
      population += children
      [point.compute_objectives() for point in population]
      population = nsga2(None, population)[:retain_size]
      gen += 1
    return population[0]


def cogee(dataset, test, train):
  model = COGEE(dataset, train).run()
  return [model.evaluate_row(row) for row in test]
