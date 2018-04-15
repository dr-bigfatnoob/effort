from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

"""
bore2.py Rule learning for multi-objective problems.
Copyright (c) 2016, Tim Menzies tim@menzies.us, MIT license v2.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

#_______________________________________________________________--
#### About

- Inspired by the Hyperband optimizer: discover good ideas
  by recursively discarding half the bad ones.
- Scores rows by their cdom score.
       - Do this only once then reuse the score.
- OUT = []
- Repeat on training data.
     - BEFORE = cdom distribution of current rows
     - Using cdom score, divide current rows into 50% best and rest.
     - Discretize numerics above and below median using the ranges in the current rows
     - Rank ranges in descending order by b^2/(b+r) where "b" is best and "r" is rest
          - TMP = upper half of the ranges
          - OUT = TMP + OUT # i.e. prepend them in sorted order
     - Discard rows that have none of TMP
     - If too few remaining rows
         exit
     - AFTER = cdom distribution of surviving rows
     - If  cliffsDelta says BEFORE == AFTER
         exit
- Report:
     - A decision ordering diagram running OUT over a test set

Note that the above incrementally discretizes, but only within zones of interest.

Todo: not linear, but clustering remaining rows and explore trees, not a line.
But not too worried about that. The SWAY experience is that most of the solutions
come from a small region.
"""

import traceback,sys,re,math,random,time,ast

# ____________________________________________________________________________________
#### Data definitions

SEP  = r"\S+"
DIRT = r'([\n\r\t]|#.*)'

# rule: function(function(x)) == function(x)
def L(x) : return math.log(float(x)) if isinstance(x,str) else x
def F(x) : return float(x)
def S(x) : return x
def I(x) : return int(x)
def X(_) : return "?"

def NUM(x): return x in [L,F,I]

def C(s,sep=SEP, dirt=DIRT):
  "Convert a string of words into a list"
  clean = re.sub(dirt, "",s)
  cells = re.findall(sep,clean)
  return [ cell.strip() for cell in cells ]

# ____________________________________________________________________________________
#### Data

# todo: if they want to optimize for recent projects, need to max year... how would that change things?

#tod:
  # make class a faracde for the data
  # add strigns as class vars
  # add a superclass that knows how to wipe and swap

def nasa93():
  return dict(
  names=[
     "recordnumber", "projectname", "cat2", "forg", "center", "year", "mode",
     "rely", "data", "cplx", "time", "stor", "virt", "turn", "acap", "aexp", "pcap", "vexp", "lexp", "modp", "tool", "sced",
     "equivphyskloc", "act_effort"],
  types=[
       X,   S,   S,                   S, S, S,    S,            S,  S,  S,  S,  S,  S, S,  S,  S,  S,  S,  S,  S,  S,  S, F,    F],
  data= [
    C("1    de   avionicsmonitoring   g  2  1979  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  25.9   117.6"),
    C("2    de   avionicsmonitoring   g  2  1979  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  24.6   117.6"),
    C("3    de   avionicsmonitoring   g  2  1979  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  7.7    31.2"),
    C("4    de   avionicsmonitoring   g  2  1979  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  8.2    36"),
    C("5    de   avionicsmonitoring   g  2  1979  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  9.7    25.2"),
    C("6    de   avionicsmonitoring   g  2  1979  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  2.2    8.4"),
    C("7    de   avionicsmonitoring   g  2  1979  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  3.5    10.8"),
    C("8    erb  avionicsmonitoring   g  2  1982  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  66.6   352.8"),
    C("9    gal  missionplanning      g  1  1980  semidetached  h   l   h   xh  xh  l  h   h   h   h   n   h   h   h   n  7.5    72"),
    C("10   gal  missionplanning      g  1  1980  semidetached  n   l   h   n   n   l  l   h   vh  vh  n   h   n   n   n  20     72"),
    C("11   gal  missionplanning      g  1  1984  semidetached  n   l   h   n   n   l  l   h   vh  h   n   h   n   n   n  6      24"),
    C("12   gal  missionplanning      g  1  1980  semidetached  n   l   h   n   n   l  l   h   vh  vh  n   h   n   n   n  100    360"),
    C("13   gal  missionplanning      g  1  1985  semidetached  n   l   h   n   n   l  l   h   vh  n   n   l   n   n   n  11.3   36"),
    C("14   gal  missionplanning      g  1  1980  semidetached  n   l   h   n   n   h  l   h   h   h   l   vl  n   n   n  100    215"),
    C("15   gal  missionplanning      g  1  1983  semidetached  n   l   h   n   n   l  l   h   vh  h   n   h   n   n   n  20     48"),
    C("16   gal  missionplanning      g  1  1982  semidetached  n   l   h   n   n   l  l   h   n   n   n   vl  n   n   n  100    360"),
    C("17   gal  missionplanning      g  1  1980  semidetached  n   l   h   n   xh  l  l   h   vh  vh  n   h   n   n   n  150    324"),
    C("18   gal  missionplanning      g  1  1984  semidetached  n   l   h   n   n   l  l   h   h   h   n   h   n   n   n  31.5   60"),
    C("19   gal  missionplanning      g  1  1983  semidetached  n   l   h   n   n   l  l   h   vh  h   n   h   n   n   n  15     48"),
    C("20   gal  missionplanning      g  1  1984  semidetached  n   l   h   n   xh  l  l   h   h   n   n   h   n   n   n  32.5   60"),
    C("21   X    avionicsmonitoring   g  2  1985  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  19.7   60"),
    C("22   X    avionicsmonitoring   g  2  1985  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  66.6   300"),
    C("23   X    simulation           g  2  1985  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  29.5   120"),
    C("24   X    monitor_control      g  2  1986  semidetached  h   n   n   h   n   n  n   n   h   h   n   n   n   n   n  15     90"),
    C("25   X    monitor_control      g  2  1986  semidetached  h   n   h   n   n   n  n   n   h   h   n   n   n   n   n  38     210"),
    C("26   X    monitor_control      g  2  1986  semidetached  n   n   n   n   n   n  n   n   h   h   n   n   n   n   n  10     48"),
    C("27   X    realdataprocessing   g  2  1982  semidetached  n   vh  h   vh  vh  l  h   vh  h   n   l   h   vh  vh  l  15.4   70"),
    C("28   X    realdataprocessing   g  2  1982  semidetached  n   vh  h   vh  vh  l  h   vh  h   n   l   h   vh  vh  l  48.5   239"),
    C("29   X    realdataprocessing   g  2  1982  semidetached  n   vh  h   vh  vh  l  h   vh  h   n   l   h   vh  vh  l  16.3   82"),
    C("30   X    communications       g  2  1982  semidetached  n   vh  h   vh  vh  l  h   vh  h   n   l   h   vh  vh  l  12.8   62"),
    C("31   X    batchdataprocessing  g  2  1982  semidetached  n   vh  h   vh  vh  l  h   vh  h   n   l   h   vh  vh  l  32.6   170"),
    C("32   X    datacapture          g  2  1982  semidetached  n   vh  h   vh  vh  l  h   vh  h   n   l   h   vh  vh  l  35.5   192"),
    C("33   X    missionplanning      g  2  1985  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  5.5    18"),
    C("34   X    avionicsmonitoring   g  2  1987  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  10.4   50"),
    C("35   X    avionicsmonitoring   g  2  1987  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  14     60"),
    C("36   X    monitor_control      g  2  1986  semidetached  h   n   h   n   n   n  n   n   n   n   n   n   n   n   n  6.5    42"),
    C("37   X    monitor_control      g  2  1986  semidetached  n   n   h   n   n   n  n   n   n   n   n   n   n   n   n  13     60"),
    C("38   X    monitor_control      g  2  1986  semidetached  n   n   h   n   n   n  n   n   n   h   n   h   h   h   n  90     444"),
    C("39   X    monitor_control      g  2  1986  semidetached  n   n   h   n   n   n  n   n   n   n   n   n   n   n   n  8      42"),
    C("40   X    monitor_control      g  2  1986  semidetached  n   n   h   h   n   n  n   n   n   n   n   n   n   n   n  16     114"),
    C("41   hst  datacapture          g  2  1980  semidetached  n   h   h   vh  h   l  h   h   n   h   l   h   h   n   l  177.9  1248"),
    C("42   slp  launchprocessing     g  6  1975  semidetached  h   l   h   n   n   l  l   n   n   h   n   n   h   vl  n  302    2400"),
    C("43   Y    application_ground   g  5  1982  semidetached  n   h   l   n   n   h  n   h   h   n   n   n   h   h   n  282.1  1368"),
    C("44   Y    application_ground   g  5  1982  semidetached  h   h   l   n   n   n  h   h   h   n   n   n   h   n   n  284.7  973"),
    C("45   Y    avionicsmonitoring   g  5  1982  semidetached  h   h   n   n   n   l  l   n   h   h   n   h   n   n   n  79     400"),
    C("46   Y    avionicsmonitoring   g  5  1977  semidetached  l   n   n   n   n   l  l   h   h   vh  n   h   l   l   h  423    2400"),
    C("47   Y    missionplanning      g  5  1977  semidetached  n   n   n   n   n   l  n   h   vh  vh  l   h   h   n   n  190    420"),
    C("48   Y    missionplanning      g  5  1984  semidetached  n   n   h   n   h   n  n   h   h   n   n   h   h   n   h  47.5   252"),
    C("49   Y    missionplanning      g  5  1980  semidetached  vh  n   xh  h   h   l  l   n   h   n   n   n   l   h   n  21     107"),
    C("50   Y    simulation           g  5  1983  semidetached  n   h   h   vh  n   n  h   h   h   h   n   h   l   l   h  78     571.4"),
    C("51   Y    simulation           g  5  1984  semidetached  n   h   h   vh  n   n  h   h   h   h   n   h   l   l   h  11.4   98.8"),
    C("52   Y    simulation           g  5  1985  semidetached  n   h   h   vh  n   n  h   h   h   h   n   h   l   l   h  19.3   155"),
    C("53   Y    missionplanning      g  5  1979  semidetached  h   n   vh  h   h   l  h   h   n   n   h   h   l   vh  h  101    750"),
    C("54   Y    missionplanning      g  5  1979  semidetached  h   n   h   h   h   l  h   n   h   n   n   n   l   vh  n  219    2120"),
    C("55   Y    utility              g  5  1979  semidetached  h   n   h   h   h   l  h   n   h   n   n   n   l   vh  n  50     370"),
    C("56   spl  datacapture          g  2  1979  semidetached  vh  h   h   vh  vh  n  n   vh  vh  vh  n   h   h   h   l  227    1181"),
    C("57   spl  batchdataprocessing  g  2  1977  semidetached  n   h   vh  n   n   l  n   h   n   vh  l   n   h   n   l  70     278"),
    C("58   de   avionicsmonitoring   g  2  1979  semidetached  h   l   h   n   n   l  l   n   n   n   n   h   h   n   l  0.9    8.4"),
    C("59   slp  operatingsystem      g  6  1974  semidetached  vh  l   xh  xh  vh  l  l   h   vh  h   vl  h   vl  vl  h  980    4560"),
    C("60   slp  operatingsystem      g  6  1975  embedded      n   l   h   n   n   l  l   vh  n   vh  h   h   n   l   n  350    720"),
    C("61   Y    operatingsystem      g  5  1976  embedded      h   n   xh  h   h   l  l   h   n   n   h   h   h   h   n  70     458"),
    C("62   Y    utility              g  5  1979  embedded      h   n   xh  h   h   l  l   h   n   n   h   h   h   h   n  271    2460"),
    C("63   Y    avionicsmonitoring   g  5  1971  organic       n   n   n   n   n   l  l   h   h   h   n   h   n   l   n  90     162"),
    C("64   Y    avionicsmonitoring   g  5  1980  organic       n   n   n   n   n   l  l   h   h   h   n   h   n   l   n  40     150"),
    C("65   Y    avionicsmonitoring   g  5  1979  embedded      h   n   h   h   n   l  l   h   h   h   n   h   n   n   n  137    636"),
    C("66   Y    avionicsmonitoring   g  5  1977  embedded      h   n   h   h   n   h  l   h   h   h   n   h   n   vl  n  150    882"),
    C("67   Y    avionicsmonitoring   g  5  1976  embedded      vh  n   h   h   n   l  l   h   h   h   n   h   n   n   n  339    444"),
    C("68   Y    avionicsmonitoring   g  5  1983  organic       l   h   l   n   n   h  l   h   h   h   n   h   n   l   n  240    192"),
    C("69   Y    avionicsmonitoring   g  5  1978  semidetached  h   n   h   n   vh  l  n   h   h   h   h   h   l   l   l  144    576"),
    C("70   Y    avionicsmonitoring   g  5  1979  semidetached  n   l   n   n   vh  l  n   h   h   h   h   h   l   l   l  151    432"),
    C("71   Y    avionicsmonitoring   g  5  1979  semidetached  n   l   h   n   vh  l  n   h   h   h   h   h   l   l   l  34     72"),
    C("72   Y    avionicsmonitoring   g  5  1979  semidetached  n   n   h   n   vh  l  n   h   h   h   h   h   l   l   l  98     300"),
    C("73   Y    avionicsmonitoring   g  5  1979  semidetached  n   n   h   n   vh  l  n   h   h   h   h   h   l   l   l  85     300"),
    C("74   Y    avionicsmonitoring   g  5  1982  semidetached  n   l   n   n   vh  l  n   h   h   h   h   h   l   l   l  20     240"),
    C("75   Y    avionicsmonitoring   g  5  1978  semidetached  n   l   n   n   vh  l  n   h   h   h   h   h   l   l   l  111    600"),
    C("76   Y    avionicsmonitoring   g  5  1978  semidetached  h   vh  h   n   vh  l  n   h   h   h   h   h   l   l   l  162    756"),
    C("77   Y    avionicsmonitoring   g  5  1978  semidetached  h   h   vh  n   vh  l  n   h   h   h   h   h   l   l   l  352    1200"),
    C("78   Y    operatingsystem      g  5  1979  semidetached  h   n   vh  n   vh  l  n   h   h   h   h   h   l   l   l  165    97"),
    C("79   Y    missionplanning      g  5  1984  embedded      h   n   vh  h   h   l  vh  h   n   n   h   h   h   vh  h  60     409"),
    C("80   Y    missionplanning      g  5  1984  embedded      h   n   vh  h   h   l  vh  h   n   n   h   h   h   vh  h  100    703"),
    C("81   hst  Avionics             f  2  1980  embedded      h   vh  vh  xh  xh  h  h   n   n   n   l   l   n   n   h  32     1350"),
    C("82   hst  Avionics             f  2  1980  embedded      h   h   h   vh  xh  h  h   h   h   h   h   h   h   n   n  53     480"),
    C("84   spl  Avionics             f  3  1977  embedded      h   l   vh  vh  xh  l  n   vh  vh  vh  vl  vl  h   h   n  41     599"),
    C("89   spl  Avionics             f  3  1977  embedded      h   l   vh  vh  xh  l  n   vh  vh  vh  vl  vl  h   h   n  24     430"),
    C("91   Y    Avionics             f  5  1977  embedded      vh  h   vh  xh  xh  n  n   h   h   h   h   h   h   n   h  165    4178.2"),
    C("92   Y    science              f  5  1977  embedded      vh  h   vh  xh  xh  n  n   h   h   h   h   h   h   n   h  65     1772.5"),
    C("93   Y    Avionics             f  5  1977  embedded      vh  h   vh  xh  xh  n  l   h   h   h   h   h   h   n   h  70     1645.9"),
    C("94   Y    Avionics             f  5  1977  embedded      vh  h   xh  xh  xh  n  n   h   h   h   h   h   h   n   h  50     1924.5"),
    C("97   gal  Avionics             f  5  1982  embedded      vh  l   vh  vh  xh  l  l   h   l   n   vl  l   l   h   h  7.25   648"),
    C("98   Y    Avionics             f  5  1980  embedded      vh  h   vh  xh  xh  n  n   h   h   h   h   h   h   n   h  233    8211"),
    C("99   X    Avionics             f  2  1983  embedded      h   n   vh  vh  vh  h  h   n   n   n   l   l   n   n   h  16.3   480"),
    C("100  X    Avionics             f  2  1983  embedded      h   n   vh  vh  vh  h  h   n   n   n   l   l   n   n   h  6.2    12")
   ])

# ______________________________________________________________________-----
#### Rows

class Row:
  """
  Rows are pairs of raw and cooked data.
  Rows know which cells are decisions and objectives.
  For the objectives, rows also know which cells need
  to minimized or maximized.
  """
  def __init__(i,raw=None):
    i.raw, i.cooked = raw, None
  def __repr__(i):
    return str(i.cooked if i.cooked else i.raw)
  def decs(i,lst): pass
  def objs(i,lst): pass
  def betters(i):  pass

class Classifier(Row):
  """
  Standard row for Classifiers. Last cell is the
  klass.
  """
  def decs(i,lst): return lst[:-1]
  def objs(i,lst): return [lst[-1]]
  def betters(i):  return [min]

class Nklass(Row):
  """
  Standard row for Moea problems.
  Rows can be compared with `cdom`.
  """
  def __init__(i,*lst,**d):
    Row.__init__(i,*lst,**d)
    i.score=0
  def cdom(i, j): # need to normalize
    def w(better):
      return -1 if better == min else 1
    def expLoss(w,x1,y1,n):
      return -1*2.71828**( w*(x1 - y1) / n )
    def loss(x, y):
      losses= []
      n = min(len(x),len(y))
      for z,bt in enumerate(i.betters()):
        x1, y1  = x[z]  , y[z]
        losses += [expLoss( w(bt),x1,y1,n)]
      return sum(losses) / n
    x = i.objs(i.cooked)
    y = j.objs(j.cooked)
    assert len(x) == len(y), "can't compare apples and oranges"
    l1= loss(x,y)
    l2= loss(y,x)
    return l1 < l2

class Coco(Nklass):
  """
  My Cocomo rows are an Moea where
  we want to max/min LOC/effort
  (which are found in the last 2 Columns.
  """
  def decs(i,lst): return lst[:-2]
  def objs(i,lst): return lst[-2:]
  def betters(i):  return [max,min]

## todo: check: can we define the standard Moea problems (e.g. fonseca) as rows?

# ______________________________________________________________________-----
#### Columns

class Column:
  """
  Columns know how to compile raw values for
  that Column, and  how to cook those values.
  They als can keep summary statistics
  for each Column.
  """
  def __init__(i,type):
    i.isDecision = True
    i.type = type
  def raw(i,x)  : return i.type(x)
  def cook(i,x) : return x

class SymColumn(Column):
  """
  Symbol Columns are nothing special.
  """
  pass

class NumColumn(Column):
  """
  Numeric Columns know how to chop values
  above and below the median value, and
  how to normalize numbers 0..1 min..max
  """
  def __init__(i,type):
    Column.__init__(i,type)
    i.lo, i.hi, i.all = 1e31, -1e31, []
    i._median = None
  def raw(i,x):
    x = i.type(x)
    i._median = None # old median now expired
    i.lo = min(i.lo,x)
    i.hi = max(i.hi,x)
    i.all += [x]
    return x
  def median(i): # maintains a cache of the median value
    if i._median is None:
      i._median= median(i.all)
    return i._median
  def cook(i,x):
    return i.chop(x) if i.isDecision else i.norm(x)
  def chop(i,x):
    return "-" if x <= i.median() else "+"
  def norm(i,x):
    return max(0,
               min(1,
                   (x - i.lo)/(i.hi - i.lo + 1e-31)))


# ______________________________________________________________________-----
#### Tables

class Table:
  """
  Tables contain Columns and rows.
  Tables organize collecting raw data, then  cook it.
  """
  def __init__(i,names= [],
               types= [],
               data=  [],
               ako =  Classifier):
    i.names = names
    i.rows  = []
    # pass0. collect meta data
    i.cols  = [ (NumColumn if NUM(t) else SymColumn)(t) for t in types ]
    for x in ako().objs(i.cols):
      x.isDecision = False
    # pass1: collect data about each Column, create "raw" rows
    for row in data:
      row    = ako([col.raw(val) for col,val in zip(i.cols,row)])
      i.rows += [row]
    # pass2: use what we know about each Column to "cook" the raw values
    for row in i.rows:
      row.cooked = [col.cook(val) for col,val in zip(i.cols,row.raw)]

class Moea(Table):
  """
  Moea Tables score each row by their cdom score.
  """
  def rankRows(i):
    "score each row according to how many other rows they dominate"
    for row1 in i.rows:
      for row2 in i.rows:
        if row1.cdom(row2):
          row1.score += 1
    i.rows = sorted(i.rows,
                    key=lambda z: z.score,
                    reverse=True)

# ______________________________________________________________________-
#### some utilities
def median(lst):
  n = len(lst)
  p = q  = n//2
  if n < 3:
    p,q = 0, n-1
  else:
    lst = sorted(lst)
    if not n % 2: q = p -1
  return lst[p] if p==q else (lst[p]+lst[q])/2

def printm(matrix,sep=","):
  "Pretty print. Columns right justified"
  s = [[str(e) for e in row] for row in matrix]
  lens = [max(map(len, col)) for col in zip(*s)]
  sep = '%s ' % sep
  fmt = sep.join('{{:>{}}}'.format(x) for x in lens)
  for row in [fmt.format(*row) for row in s]:
    print(row)

def literal(x):
  try:
    return ast.literal_eval(x)
  except Exception:
    return x

def comLine2Dictionary():
  d,pairs={},[]
  for x in sys.argv[2:]:
    if   x[0] == "-": d[re.sub('^-*',"",x)] = False
    elif x[0] == "+": d[re.sub('^\+*',"",x)] = True
    else            : pairs += [x]
  str= ' '.join(pairs)
  pat= re.compile(r'(\S+)=([^ ]+)[ $]*')
  d.update({key:literal(val) for (key,val) in re.findall(pat,str) })
  return d


# ______________________________________________________________________-
#### demo stuff

def eg(f=None,want=None,dic={},lst=[], all={},names=[]):
  "Decorator for functions that can be called from command line."
  if want=="help":
    for name in names:
      doc = all[name].__doc__
      if doc:
        print(name, "\t: ",re.sub(r'\n[ \t]*',"\n ",doc))
    return print("help","\t: ","print this help text")
  if want: # run one example
    if not want in all:
      return print("# cannot execute: missing %s" % want)
    f=all[want]
    hdr = "\n-----| %s |"+ ("-"*40)
    print(hdr % f.__name__,end="\n# ")
    if f.__doc__:
      print(re.sub(r'\n[ \t]*',"\n# ",f.__doc__))
    print("")
    # t1=time.process_time()
    t1=time.time()
    f(*lst,**dic)
    # t2=time.process_time()
    t2=time.time()
    print("# pass","(%.4f secs)" % (t2-t1))
  else:
    if f: # add one example
      all[f.__name__] = f
      names += [f.__name__]
    else: # run all examples, count how many do not crash
      n=y=0
      for name in names:
        try:
          eg(want=name)
          y += 1
        except Exception:
          n += 1
          print(traceback.format_exc())
      print("# tried= ",y+n," %passed= ",100*round(y/(y+n)))

### and here are the demos that can be called at the command line

# @eg
# def eg0():
#   "basic test, simple classifier"
#   t = Table(**nasa93())
#   printm([row.cooked for row in t.rows])
#   print(t.rows[-4].raw)
#   print(t.rows[-4].cooked)

@eg
def eg1():
  "can we handle multi-obj?"
  t = Moea(ako=Coco,**nasa93())
  t.rankRows()
  printm([row.cooked for row in t.rows])


# ______________________________________________________________________-
#### main

if __name__ ==  "__main__":
  if len(sys.argv) > 1 and sys.argv[1]:
    eg(want=sys.argv[1],
       dic=comLine2Dictionary())
  else:
    eg()
