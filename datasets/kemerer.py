from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"


from dataset import Dataset, Meta, read_csv


class Kemerer(Dataset):
  def __init__(self):
    Dataset.__init__(self, data=Kemerer.data(),
                     dec_meta=Kemerer.decision_meta(),
                     obj_meta=Kemerer.objective_meta())

  @staticmethod
  def decision_meta():
    return [Meta(index=0, name="Language", type=Meta.DISC, range=(1, 4), is_obj=False),
            Meta(index=1, name="Hardware", type=Meta.DISC, range=(1, 7), is_obj=False),
            Meta(index=2, name="Duration", type=Meta.CONT, range=(5, 8), is_obj=False),
            Meta(index=3, name="KSLOC", type=Meta.CONT, range=(39, 451), is_obj=False),
            Meta(index=4, name="AdjFp", type=Meta.CONT, range=(99.9, 2307.8), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=5, name="EffortMM", type=Meta.CONT, range=(23.2, 1108.31), is_obj=False)]

  @staticmethod
  def data():
    return read_csv("datasets/csv/kemerer.arff.csv")