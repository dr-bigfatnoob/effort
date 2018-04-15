from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"


from dataset import Dataset, Meta, read_csv


class ISBSG10(Dataset):
  def __init__(self):
    Dataset.__init__(self, data=ISBSG10.data(),
                     dec_meta=ISBSG10.decision_meta(),
                     obj_meta=ISBSG10.objective_meta())

  @staticmethod
  def decision_meta():
    return [Meta(index=0, name="Data_Quality", type=Meta.DISC, range=(1, 3), is_obj=False),
            Meta(index=1, name="UFP", type=Meta.DISC, range=(1, 3), is_obj=False),
            Meta(index=2, name="IS", type=Meta.DISC, range=(1, 11), is_obj=False),
            Meta(index=3, name="DP", type=Meta.DISC, range=(1, 6), is_obj=False),
            Meta(index=4, name="LT", type=Meta.DISC, range=(1, 4), is_obj=False),
            Meta(index=5, name="PPL", type=Meta.DISC, range=(1, 15), is_obj=False),
            Meta(index=6, name="CA", type=Meta.DISC, range=(1, 3), is_obj=False),
            Meta(index=7, name="FS", type=Meta.CONT, range=(44, 1372), is_obj=False),
            Meta(index=8, name="RS", type=Meta.DISC, range=(1, 5), is_obj=False),
            Meta(index=9, name="Recording_Method", type=Meta.DISC, range=(1, 5), is_obj=False),
            Meta(index=10, name="FPS", type=Meta.DISC, range=(1, 6), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=11, name="Effort", type=Meta.CONT, range=(87, 14454), is_obj=True)]

  @staticmethod
  def data():
    return read_csv("datasets/csv/isbsg10.arff.csv")
