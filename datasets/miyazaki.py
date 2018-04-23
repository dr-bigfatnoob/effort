from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from dataset import Dataset, Meta, read_csv


class Miyazaki(Dataset):
  def __init__(self):
    Dataset.__init__(self, data=Miyazaki.data(),
                     dec_meta=Miyazaki.decision_meta(),
                     obj_meta=Miyazaki.objective_meta())

  @staticmethod
  def decision_meta():
    return [Meta(index=0, name="SCRN", type=Meta.CONT, range=(0, 281), is_obj=False),
            Meta(index=1, name="FORM", type=Meta.CONT, range=(0, 91), is_obj=False),
            Meta(index=2, name="FILE", type=Meta.CONT, range=(2, 370), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=3, name="Effort", type=Meta.CONT, range=(896, 253760), is_obj=True)]

  @staticmethod
  def data():
    return read_csv("datasets/csv/miyazaki.arff.csv")