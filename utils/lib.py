from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

import random

# Constants
EPS = 0.000001


# Utility Functions
def say(*lst):
  print(*lst, end="")
  sys.stdout.flush()


def gt(a, b): return a > b


def lt(a, b): return a < b


def gte(a, b): return a >= b


def lte(a, b): return a <= b


def ne(a, b): return a != b


def eq(a, b): return a == b


def median_iqr(lst, ordered=False):
  if not ordered:
    lst = sorted(lst)
  n = len(lst)
  q = n // 4
  iqr = lst[q * 3] - lst[q]
  if n % 2:
    return lst[q * 2], iqr
  else:
    p = max(0, q - 1)
    return (lst[p] + lst[q]) * 0.5, iqr


def uniform(low, high):
  return random.uniform(low, high)


# Utility Classes
class O:
  """
  Basic Class. All classes in this project
  should directly or indirectly extend this class
  """
  def __init__(self, **d):
    self.has().update(**d)

  def has(self):
    return self.__dict__

  def update(self, **d):
    self.has().update(d)
    return self

  def __repr__(self):
    show = [':%s %s' % (k, self.has()[k])
            for k in sorted(self.has().keys())
            if k[0] is not "_"]
    txt = ' '.join(show)
    if len(txt) > 60:
      show = map(lambda x: '\t' + x + '\n', show)
    return '{' + ' '.join(show) + '}'

  def has_attr(self, attr):
    return getattr(self, attr, None) is not None


class N(O):
  """
  Add/delete counts of numbers.
  """
  def __init__(self, inits=None):
    O.__init__(self)
    self.n = self.mu = self.m2 = 0
    self.cache = Cache()
    if inits is None: inits = []
    map(self.__iadd__, inits)

  def zero(self):
    self.n = self.mu = self.m2 = 0
    self.cache = Cache()

  def sd(self):
    if self.n < 2:
      return 0
    else:
      return (max(0, self.m2) / (self.n - 1))**0.5

  def __iadd__(self, x):
    self.cache += x
    self.n += 1
    delta = x - self.mu
    self.mu += delta / (1.0 * self.n)
    self.m2 += delta * (x - self.mu)
    return self

  def __isub__(self, x):
    self.cache = Cache()
    if self.n < 2: return self.zero()
    self.n -= 1
    delta = x - self.mu
    self.mu -= delta / (1.0 * self.n)
    self.m2 -= delta * (x - self.mu)
    return self


class Cache:
  """
  Keep a random sample of stuff seen so far.
  """
  max_size = 128

  def __init__(self, inits=None):
    self.all, self.n, self._has = [], 0, None
    if inits is None: inits = []
    map(self.__iadd__, inits)

  def __iadd__(self, x):
    self.n += 1
    if len(self.all) < Cache.max_size:  # if not full
      self._has = None
      self.all += [x]               # then add
    else:  # otherwise, maybe replace an old item
      if random.random() <= Cache.max_size / self.n:
        self._has = None
        self.all[int(random.random() * Cache.max_size)] = x
    return self

  def has(self):
    if self._has is None:
      lst = sorted(self.all)
      med, iqr = median_iqr(lst, ordered=True)
      self._has = O(median=med, iqr=iqr,
                    lo=self.all[0], hi=self.all[-1])
    return self._has
