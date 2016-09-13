from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from dataset import Dataset, Meta, read_csv


class Desharnais(Dataset):
  def __init__(self):
    Dataset.__init__(self, data=Desharnais.data(),
                     dec_meta=Desharnais.decision_meta(),
                     obj_meta=Desharnais.objective_meta())

  @staticmethod
  def decision_meta():
    return [Meta(index=0, name="TeamExp", type=Meta.CONT, range=(0, 4), is_obj=False),
            Meta(index=1, name="ManagerExp", type=Meta.CONT, range=(0, 7), is_obj=False),
            Meta(index=2, name="Transactions", type=Meta.CONT, range=(9, 886), is_obj=False),
            Meta(index=3, name="Entities", type=Meta.CONT, range=(7, 387), is_obj=False),
            Meta(index=4, name="PointsAdjust", type=Meta.CONT, range=(73, 1127), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=5, name="Effort", type=Meta.CONT, range=(546, 23940), is_obj=True)]

  @staticmethod
  def data():
    return read_csv("datasets/csv/desharnais.arff.csv")
