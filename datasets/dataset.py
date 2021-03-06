from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from utils.lib import *
import csv
import numpy as np


class Meta(O):
  CONT = "continuous"
  DISC = "discrete"

  def __init__(self, **d):
    """
    Initialize metadata for a dataset
    :param index:
    :param name:
    :param type: (Meta.cont, Meta.desc)
    :param range: (If Meta.cont (min, max))
    :param is_obj: True/False
    """
    O.__init__(self, **d)


class Dataset(O):
  def __init__(self, data, **d):
    """
    :param data
    :param d: dec_meta = metadata list for decisions
              obj_meta = metadata list for objectives

    """
    if 'dec_meta' in d and 'obj_meta' in d:
      rows = []
      for line in data:
        row = []
        for i, (meta, cell) in enumerate(zip(d['dec_meta']+d['obj_meta'], line)):
          if meta.type == Meta.CONT:
            row.append(float(cell))
          else:
            row.append(str(int(cell)))
        rows.append(O(cells=row))
    else:
      rows = [O(cells=row) for row in data]
    O.__init__(self, _rows=rows, _distances={}, **d)

  def get_rows(self):
    return self._rows

  def get_distances(self):
    return self._distances

  def get_distance(self, one, two):
    distances = self.get_distances().get(one.id, None)
    if distances is None: return None
    return distances.get(two.id, None)

  def set_distance(self, one, two, distance):
    distances = self.get_distances().get(one.id, {})
    distances[two.id] = distance
    self.get_distances()[one.id] = distances
    distances = self.get_distances().get(two.id, {})
    distances[one.id] = distance
    self.get_distances()[two.id] = distances

  def effort(self, row):
    return row.cells[self.obj_meta[0].index]

  def effort_variance(self, rows):
    if rows is None or len(rows) == 0:
      return np.inf
    return np.var([self.effort(row) for row in rows])


def read_csv(file_path, read_header=False):
  lst = []
  with open(file_path) as f:
    csv_f = csv.reader(f)
    for row in csv_f:
      lst.append(row)
  if read_header:
    return lst
  return lst[1:]


def read_pandas_dataframe(df, read_header=False):
  headers = list(df.columns)
  values = df.values.tolist()
  rows = []
  for row in values:
    rows.append(map(parse_string, row))
  if read_header:
    return [headers] + rows
  return values


def parse_string(s):
  try:
    return int(s)
  except ValueError:
    try:
      return float(s)
    except ValueError:
      return s
