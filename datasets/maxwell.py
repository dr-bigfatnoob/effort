from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from dataset import Dataset, Meta, read_csv


class Maxwell(Dataset):
  def __init__(self):
    Dataset.__init__(self, data=Maxwell.data(),
                     dec_meta=Maxwell.decision_meta(),
                     obj_meta=Maxwell.objective_meta())

  @staticmethod
  def decision_meta():
    return [Meta(index=0, name="T01", type=Meta.CONT, range=(1, 5), is_obj=False),
            Meta(index=1, name="T02", type=Meta.CONT, range=(1, 5), is_obj=False),
            Meta(index=2, name="T03", type=Meta.CONT, range=(2, 5), is_obj=False),
            Meta(index=3, name="T04", type=Meta.CONT, range=(2, 5), is_obj=False),
            Meta(index=4, name="T05", type=Meta.CONT, range=(1, 5), is_obj=False),
            Meta(index=5, name="T06", type=Meta.CONT, range=(1, 4), is_obj=False),
            Meta(index=6, name="T07", type=Meta.CONT, range=(1, 5), is_obj=False),
            Meta(index=7, name="T08", type=Meta.CONT, range=(2, 5), is_obj=False),
            Meta(index=8, name="T09", type=Meta.CONT, range=(2, 5), is_obj=False),
            Meta(index=9, name="T10", type=Meta.CONT, range=(2, 5), is_obj=False),
            Meta(index=10, name="T11", type=Meta.CONT, range=(2, 5), is_obj=False),
            Meta(index=11, name="T12", type=Meta.CONT, range=(2, 5), is_obj=False),
            Meta(index=12, name="T13", type=Meta.CONT, range=(1, 5), is_obj=False),
            Meta(index=13, name="T14", type=Meta.CONT, range=(1, 5), is_obj=False),
            Meta(index=14, name="T15", type=Meta.CONT, range=(1, 5), is_obj=False),
            Meta(index=15, name="NLAN", type=Meta.CONT, range=(1, 4), is_obj=False),
            Meta(index=16, name="Duration", type=Meta.CONT, range=(4, 54), is_obj=False),
            Meta(index=17, name="Size", type=Meta.CONT, range=(48, 3643), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=18, name="Effort", type=Meta.CONT, range=(583, 63694), is_obj=True)]

  @staticmethod
  def data():
    return read_csv("datasets/csv/maxwell.arff.csv")
