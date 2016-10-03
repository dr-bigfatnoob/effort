from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.errors import absolute_errors, confidence
from nsga2 import *
from methods.teak import teak


class TeakProblem(Problem):
  def __init__(self, dataset, train, tune, test):
    decisions = [Decision("parent_factor", 0.5, 2.0), Decision("max_factor", 0.25, 1.0)]
    objectives = [Objective("error", do_minimize=True), Objective("confidence", do_minimize=True)]
    Problem.__init__(self, dataset, train, tune, decisions, objectives)
    self.test = test

  def evaluate(self, point):
    if not point.objectives:
      params = {meta.name: val for meta, val in zip(self.decisions, point.decisions)}
      predicted = teak(self.dataset, self.tune, self.train, **params)
      actuals = [self.dataset.effort(row) for row in self.tune]
      errors = absolute_errors(actuals, predicted)
      sae = sum(errors)
      conf = confidence(errors, len(self.tune), len(self.decisions) + 1, 0.95)
      point.objectives = [sae, conf]
    return point.objectives

  def predict(self, point):
    params = {meta.name: val for meta, val in zip(self.decisions, point.decisions)}
    return teak(self.dataset, self.test, self.train, **params)

  def check_constraints(self, point):
    if self.objectives is not None: return True
    params = {meta.name: val for meta, val in zip(self.decisions, point.decisions)}
    predicted = teak(self.dataset, self.tune, self.train, **params)
    if predicted is None:
      return False
    actuals = [self.dataset.effort(row) for row in self.tune]
    errors = absolute_errors(actuals, predicted)
    sae = sum(errors)
    conf = confidence(errors, len(self.tune), len(self.decisions) + 1, 0.95)
    point.objectives = [sae, conf]
    return True


class TeakOptimize(O):
  def __init__(self, problem):
    O.__init__(self)
    self.problem = problem

  def crossover_mutate(self, mom, dad):
    """
    Perform crossover and mutation
    :param mom: Parent 1
    :param dad: Parent 2
    :param cr: Crossover rate
    :param mr: Mutation rate
    :return: [bro, sis]
    """
    bro, sis = sbx(self.problem, mom, dad)
    bro = poly_mutate(self.problem, bro)
    sis = poly_mutate(self.problem, sis)
    return [bro, sis]

  def run(self, pop_size=100, gens=10, retain_size=10):
    gen = 0
    population = self.problem.populate(pop_size)
    [self.problem.evaluate(point) for point in population]
    while gen < gens:
      population = nsga2(self.problem, population)
      roulette_map = make_roulette_map(population)
      children = []
      for _ in xrange(len(population)):
        [mom, dad] = roulette_wheel(roulette_map, 2)
        kids = self.crossover_mutate(mom, dad)
        if kids is None:
          continue
        for kid in kids:
          if self.problem.check_constraints(kid):
            self.problem.evaluate(kid)
            children.append(kid)

      population += children
      [self.problem.evaluate(point) for point in population]
      population = nsga2(self.problem, population)[:retain_size]
      gen += 1
    return population


def _main():
  from datasets.albrecht import Albrecht
  from datasets.maxwell import Maxwell
  from utils.validation import kfold
  dataset = Maxwell()
  for test, rest in kfold(dataset.get_rows(), 3):
    cut = int(0.75 * len(rest))
    train, tune = rest[:cut], rest[cut:]
    print(len(train), len(tune), len(test))
    problem = TeakProblem(dataset, train, tune, test)
    optimizer = TeakOptimize(problem)
    best = optimizer.run()
    actuals = [problem.dataset.effort(row) for row in test]
    pred1 = problem.predict(best[0])
    pred2 = problem.predict(best[-1])
    err1 = sum(absolute_errors(actuals, pred1))
    err2 = sum(absolute_errors(actuals, pred2))
    print(err1, err2)
    exit()

if __name__ == "__main__":
  _main()
