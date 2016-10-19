from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import *
from datasets.dataset import Meta


class Node(O):
  def __init__(self, rows, parent=None, variance=None):
    O.__init__(self)
    self._rows = rows
    self.parent = parent
    self.kids = []
    self.variance = variance

  def get_rows(self):
    return self._rows


def default_settings(**params):
  return O(prefix="|.. ",
           min_size=10,
           depth_min=2,
           depth_max=10,
           do_prune=True,
           verbose=False,
           east_west=east_west_slow
           ).update(**params)


def normalize(dataset, index, value):
  return (value - dataset.dec_meta[index].range[1]) / \
         (dataset.dec_meta[index].range[1] - dataset.dec_meta[index].range[0] + EPS)


def distance(dataset, one, two):
  dist = dataset.get_distance(one, two)
  if dist is not None:
    return dist
  n = len(one.cells)
  dist = 0
  for index, meta in enumerate(dataset.dec_meta):
    if meta.type == Meta.CONT:
      n1 = normalize(dataset, index, one.cells[index])
      n2 = normalize(dataset, index, two.cells[index])
      dist += (n1 - n2)**2
    else:
      if one.cells[index] != two.cells[index]:
        dist += 1
  dist = (dist / n)**0.5
  dataset.set_distance(one, two, dist)
  return dist


def furthest(dataset, row, rows, init=0, better=gt):
  far, d = row, init
  for one in rows:
    if row == one: continue
    temp = distance(dataset, row, one)
    if better(temp, d): far, d = one, temp
  return far


def closest(dataset, row, rows):
  return furthest(dataset, row, rows,
                  init=sys.maxint, better=lt)


def closest_n(dataset, row, rows, n):
  lst = []
  for one in rows:
    if id(one) == id(row): continue
    d = distance(dataset, row, one)
    lst += [(d, one)]
  return sorted(lst)[:n]


def east_west_slow(dataset, rows):
  max_dist = -sys.maxint
  east_index, west_index = None, None
  for i in range(len(rows) - 1):
    for j in range(i + 1, len(rows)):
      temp = distance(dataset, rows[i], rows[j])
      if temp > max_dist:
        max_dist, east_index, west_index = temp, i, j
  return rows[east_index], rows[west_index]


def east_west_fast(dataset, rows):
  one = random.choice(rows)
  west = furthest(dataset, one, rows)
  east = furthest(dataset, west, rows)
  return east, west


def max_tree_variance(root):
  max_variance = -1
  for node, level in walk_tree(root):
    max_variance = max(max_variance, node.variance)
  return max_variance


def fastmap(dataset, rows, settings):
  east, west = settings.east_west(dataset, rows)
  c = distance(dataset, east, west)
  lst = []
  for one in rows:
    a = distance(dataset, one, west)
    b = distance(dataset, one, east)
    if c == 0:
      x = 0
    else:
      x = (a ** 2 + c ** 2 - b ** 2) / (2 * c)
    lst += [(x, one)]
  lst = sorted(lst)
  mid = len(lst) // 2
  wests = map(lambda pt: pt[1], lst[:mid])
  easts = map(lambda pt: pt[1], lst[mid:])
  west = wests[0]
  east = easts[-1]
  return wests, west, easts, east, lst[mid][0]


def where(dataset, rows, level=0, parent=None, settings=None):
  if not settings:
    settings = default_settings()

  def too_deep():
    return level > settings.depth_max

  def too_few():
    return len(rows) < settings.min_size

  def show(suffix):
    if settings.verbose:
      print(settings.prefix * level, len(rows), suffix, ';', id(node) % 1000, sep='')

  node = Node(rows, parent, variance=dataset.effort_variance(rows))
  if too_deep() or too_few():
    if settings.verbose:
      show(".")
  else:
    if settings.verbose:
      show("")
    wests, west, easts, east, mid = fastmap(dataset, rows, settings)
    node.update(east=east, west=west, mid=mid)
    node.kids += [where(dataset, wests, level + 1, node, settings)]
    node.kids += [where(dataset, easts, level + 1, node, settings)]
  return node


def get_leaf(dataset, row, node):
  if len(node.kids) > 1:
    east, west, mid = node.east, node.west, node.mid
    a = distance(dataset, row, west)
    b = distance(dataset, row, east)
    c = distance(dataset, west, east)
    if c == 0: return node
    x = (a**2 + c**2 - b**2) / (2 * c)
    if x < mid:
      return get_leaf(dataset, row, node.kids[0])
    else:
      return get_leaf(dataset, row, node.kids[1])
  elif len(node.kids) == 1:
    return get_leaf(dataset, row, node.kids[0])
  return node


def walk_tree(node, visited=None, level=0):
  if visited is None: visited = set([])
  if node:
    if not id(node) in visited:
      visited.add(id(node))
      yield node, level
      for kid in node.kids:
        for sub_node, sub_level in walk_tree(kid, visited, level + 1):
          yield sub_node, sub_level


def get_leaves(node):
  for node, level in walk_tree(node):
    if not node.kids:
      yield node, level


def build(dataset, rows=None, **params):
  if rows is None:
    rows = dataset.get_rows()
  settings = default_settings(**params).update(min_size=2 * int(len(rows)**0.5))
  return where(dataset, rows, settings=settings)


if __name__ == "__main__":
  from datasets.albrecht import Albrecht
  build(Albrecht(), verbose=True)
