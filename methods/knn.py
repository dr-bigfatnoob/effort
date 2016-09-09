from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from methods.where import closest, closest_n

__author__ = 'panzer'


def knn_1(dataset, test, train):
  predicts = []
  for one in test:
    closest_1 = closest(dataset, one, train)
    predicts.append(dataset.effort(closest_1))
  return predicts


def knn_3(dataset, test, train):
  predicts = []
  for one in test:
    closest_3 = closest_n(dataset, one, train, 3)
    a = dataset.effort(closest_3[0][1])
    b = dataset.effort(closest_3[1][1])
    c = dataset.effort(closest_3[2][1])
    predicts.append((3 * a + 2 * b + 1 * c) / 6)
  return predicts
