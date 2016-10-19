from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from dataset import Dataset, Meta, read_csv


class Finnish(Dataset):
  def __init__(self):
    Dataset.__init__(self, data=Finnish.data(),
                     dec_meta=Finnish.decision_meta(),
                     obj_meta=Finnish.objective_meta())

  @staticmethod
  def decision_meta():
    return [Meta(index=0, name='hw', type=Meta.DISC, values=range(1, 4), is_obj=False),
            Meta(index=1, name='at', type=Meta.DISC, values=range(1, 6), is_obj=False),
            Meta(index=2, name='FP', type=Meta.CONT, range=(65, 1814), is_obj=False),
            Meta(index=3, name='co', type=Meta.DISC, values=range(2, 11), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=4, name="Effort", type=Meta.CONT, range=(6.13, 10.2), is_obj=True)]

  @staticmethod
  def data():
    return read_csv("datasets/csv/finnish.arff.csv")
