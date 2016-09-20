from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import *
from math import sqrt
from collections import OrderedDict
import numpy as np
import scipy.stats as stats
import bisect


def rand():
  return random.uniform(0, 1e3)


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


def loo(points):
  for i in range(len(points)):
    yield points[i], points[:i] + points[i+1:]


def nsga2(points):
  frontiers = []
  front1 = []
  for one, rest in loo(points):
    one.dominated = []
    one.dominating = 0
    for two in rest:
      if one.dominates(two):
        one.dominated.append(two)
      elif two.dominates(one):
        one.dominating += 1
    if one.dominating == 0:
      front1.append(one)
  frontiers.append(assign_crowd_distance(front1))
  while True:
    front2 = []
    for one in front1:
      for two in one.dominated:
        two.dominating -= 1
        if two.dominating == 0:
          front2.append(two)
    if len(front2) == 0:
      break
    frontiers.append(assign_crowd_distance(front2))
    front1 = front2
  i = 0
  total = len(points)
  sorted_points = []
  for front in frontiers:
    for point in front:
      point.score = 2 * (total - i) / (total * (total + 1))
      sorted_points.append(point)
      i += 1
  return sorted_points


def assign_crowd_distance(frontier):
  """
  Crowding distance between each point in frontier
  :param frontier: List of points
  :return:
  """
  l = len(frontier)
  assert l > 0
  for m in range(len(frontier[0].objectives)):
    frontier = sorted(frontier, key=lambda x: x.objectives[m])
    frontier[0].crowd_distance = float("inf")
    frontier[-1].crowd_distance = float("inf")
    for i in range(1, l-1):
      frontier[i].crowd_distance += frontier[i+1].objectives[m] - frontier[i-1].objectives[m]
  return sorted(frontier, key=lambda x: x.crowd_distance)


def make_roulette_map(population, is_sorted=True):
  if not sorted:
    population = sorted(population, key=lambda x:x.score, reverse=True)
  r_map = OrderedDict()
  cum = 0.0
  for point in population:
    cum += point.score
    r_map[cum] = point
  return r_map


def roulette_wheel(roulette_map, number):
  chosen = []
  for _ in xrange(number):
    r = random.random()
    index = bisect.bisect(roulette_map.keys(), r)
    chosen.append(roulette_map[roulette_map.keys()[index - 1]])
  return chosen


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
    else: raise RuntimeError("Invalid symbol : %s"%sel)



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
    while r==self.value:
      r = rand()
    return Constant(rand())


class Variable(O):
  def __init__(self, name):
    O.__init__(self, name=name)

  def mutate(self):
    return self


class Point(O):
  def __init__(self, sequence, dataset, rows, better=(lt, lt)):
    O.__init__(self)
    self.sequence = sequence
    self.objectives = None
    self._dataset = dataset
    self._rows = rows
    self.better = better
    self.dominating = 0
    self.dominated = []
    self.crowd_distance = 0.0
    self.score = 0.0

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
    new = Point(self.sequence, self._dataset, self._rows)
    new.objectives = self.objectives
    return new

  def dominates(self, another):
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
    return Point(sequence, self.dataset, self.rows)

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
      bro = Point(mom.sequence[:split] + dad.sequence[split:], self.dataset, self.rows)
      sis = Point(dad.sequence[:split] + mom.sequence[split:], self.dataset, self.rows)
      for i in xrange(l):
        r = random.random()
        if r < random.random():
          bro.sequence[i] = bro.sequence[i].mutate()
        r = random.random()
        if r < random.random():
          sis.sequence[i] = sis.sequence[i].mutate()
      return [bro, sis]
    else:
      return None

  def run(self, pop_size=100, gens=250, retain_size=10):
    gen = 0
    population = self.populate(pop_size)
    [point.compute_objectives() for point in population]
    while gen < gens:
      # say(len(population), ".")
      population = nsga2(population)
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
      population = nsga2(population)[:retain_size]
      gen += 1
    return population[0]


def cogee(dataset, test, train):
  model = COGEE(dataset, train).run()
  return [model.evaluate_row(row) for row in test]