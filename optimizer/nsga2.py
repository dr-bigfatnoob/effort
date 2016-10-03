from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import *
from collections import OrderedDict
import bisect

class Decision(O):
  """
  Meta Data for a Decision
  """
  def __init__(self, name, low, high):
    O.__init__(self)
    self.name = name
    self.low = low
    self.high = high

  def normalize(self, value):
    return (value - self.low) / (self.high - self.low)


class Objective(O):
  """
  Meta Data for an objective
  """
  def __init__(self, name, do_minimize=True, low=0, high=1):
    O.__init__(self)
    self.do_minimize = do_minimize
    self.name = name
    self.low = low
    self.high = high
    self.better = lt if self.do_minimize else gt


class Problem(O):
  def __init__(self, dataset, train, tune, decisions, objectives):
    O.__init__(self)
    self.dataset = dataset
    self.train = train
    self.tune = tune
    self.decisions = decisions
    self.objectives = objectives

  def evaluate(self, point):
    assert False

  def check_constraints(self, point):
    return True

  def generate_one(self):
    while True:
      one = NSGAPoint([uniform(d.low, d.high) for d in self.decisions])
      if self.check_constraints(one):
        return one

  def populate(self, size):
    return [self.generate_one() for _ in xrange(size)]


class NSGAPoint(O):
  def __init__(self, decisions):
    """
    Initialize a Point in the Optimizer
    :param decisions: List of decisions
    """
    O.__init__(self)
    self.decisions = decisions
    self.objectives = None
    self.rank = 0
    self.dominated = []
    self.dominating = 0
    self.crowd_distance = 0
    self.score = 0.0

  def clone(self):
    """
    Clone an instance of point
    :return:
    """
    new = NSGAPoint(self.decisions)
    if self.objectives:
      new.objectives = self.objectives[:]
    return new

  def clear(self):
    self.rank = 0
    self.dominated = []
    self.dominating = 0
    self.crowd_distance = 0

  def dominates(self, another, problem):
    assert self.objectives is not None
    assert another.objectives is not None
    better = False
    for i, (one, two) in enumerate(zip(self.objectives, another.objectives)):
      if problem.objectives[i].better(one, two):
        better = True
      elif one != two:
        return False
    return better

  def __eq__(self, other):
    return self.decisions == other.decisions

  def __hash__(self):
    return hash(frozenset(self.decisions))


def loo(points):
  for i in range(len(points)):
    yield points[i], points[:i] + points[i+1:]


def nsga2(problem, points):
  """
  Perform nsga2 sorting
  :param points: List of Points. Must have an attribute called objectives which
  is a a list of float
  :return:
  """
  frontiers = []
  front1 = []
  for one, rest in loo(points):
    one.dominated = []
    one.dominating = 0
    for two in rest:
      if one.dominates(two, problem):
        one.dominated.append(two)
      elif two.dominates(one, problem):
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


SBX_CR = 1
SBX_ETA = 30
PM_ETA = 20


def get_beta_q(rand, alpha, eta=30):
  if rand <= (1.0/alpha):
    return (rand * alpha) ** (1.0/(eta+1.0))
  else:
    return (1.0/(2.0 - rand*alpha)) ** (1.0/(eta+1.0))


def sbx(problem, mom, dad, **params):
  """
  Simulated Binary Crossover
  :param problem: Instance of problem
  :param mom: Instance of NSGAPoint
  :param dad: Instance of NSGAPoint
  :param params:
  :return: sis, bro
  """
  cr = params.get("cr", SBX_CR)
  eta = params.get("eta", SBX_ETA)
  sis_decs = mom.decisions[:]
  bro_decs = dad.decisions[:]
  if random.random() > cr: return mom, dad
  for i, decision in enumerate(problem.decisions):
    if random.random() > 0.5:
      sis_decs[i], bro_decs[i] = bro_decs[i], sis_decs[i]
      continue
    if abs(sis_decs[i] - bro_decs[i]) <= EPS:
      continue
    low = problem.decisions[i].low
    high = problem.decisions[i].high
    small = min(sis_decs[i], bro_decs[i])
    large = max(sis_decs[i], bro_decs[i])
    some = random.random()

    beta = 1.0 + (2.0 * (small - low) / (large - small))
    alpha = 2.0 - beta ** (-1 * (eta + 1.0))
    beta_q = get_beta_q(some, alpha)
    sis_decs[i] = 0.5 * ((small + large) - beta_q * (large - small))
    sis_decs[i] = max(low, min(sis_decs[i], high))

    beta = 1.0 + (2.0 * (high - large) / (large - small))
    alpha = 2.0 - beta ** (-1 * (eta + 1.0))
    beta_q = get_beta_q(some, alpha)
    bro_decs[i] = 0.5 * ((small + large) + beta_q * (large - small))
    bro_decs[i] = max(low, min(bro_decs[i], high))
    if random.random() > 0.5:
      sis_decs[i], bro_decs[i] = bro_decs[i], sis_decs[i]
  return NSGAPoint(sis_decs), NSGAPoint(bro_decs)


def poly_mutate(problem, one, **params):
  """
  Polynomial Mutation
  :param problem: Instance of problem
  :param one: Instance of point
  :param params:
  :return:
  """
  pm = params.get("pm", 1 / len(problem.decisions))
  eta = params.get("eta", PM_ETA)
  mutant = [0] * len(problem.decisions)
  one_decs = one.decisions
  for i, decision in enumerate(problem.decisions):
    if random.random() > pm:
      mutant[i] = one_decs[i]
      continue
    low = problem.decisions[i].low
    high = problem.decisions[i].high
    del1 = (one_decs[i] - low)/(high - low)
    del2 = (high - one_decs[i])/(high - low)
    mut_pow = 1 / (eta + 1)
    rand_no = random.random()
    if rand_no < 0.5:
      xy = 1 - del1
      val = 2 * rand_no + (1 - 2 * rand_no) * (xy ** (eta + 1))
      del_q = val ** mut_pow - 1
    else:
      xy = 1 - del2
      val = 2 * (1 - rand_no) + 2 * (rand_no - 0.5) * (xy ** (eta + 1))
      del_q = 1 - val ** mut_pow
    mutant[i] = max(low, min(one_decs[i] + del_q * (high - low), high))
  return NSGAPoint(mutant)


def make_roulette_map(population, is_sorted=True):
  """
  Make a roulette Map for population
  :param population:
  :param is_sorted:
  :return:
  """
  if not is_sorted:
    population = sorted(population, key=lambda x: x.score, reverse=True)
  r_map = OrderedDict()
  cum = 0.0
  for point in population:
    cum += point.score
    r_map[cum] = point
  return r_map


def roulette_wheel(roulette_map, number):
  """
  Select Wheel
  :param roulette_map: Instance of Roulette Map
  :param number: Number of instances to select
  :return:
  """
  chosen = []
  for _ in xrange(number):
    r = random.random()
    index = bisect.bisect(roulette_map.keys(), r)
    chosen.append(roulette_map[roulette_map.keys()[index - 1]])
  return chosen
