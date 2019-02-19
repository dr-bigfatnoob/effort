from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from datasets.dataset import Dataset, Meta, read_csv
from datasets.cleaned import data_to_use


class Finnish(Dataset):
  def __init__(self):
    Dataset.__init__(self, data=Finnish.data(),
                     dec_meta=Finnish.decision_meta(),
                     obj_meta=Finnish.objective_meta())

  @staticmethod
  def decision_meta():
    return [Meta(index=0, name='HW', type=Meta.DISC, values=range(1, 4), is_obj=False),
            Meta(index=1, name='AT', type=Meta.DISC, values=range(1, 6), is_obj=False),
            Meta(index=2, name='FP', type=Meta.CONT, range=(65, 1814), is_obj=False),
            Meta(index=3, name='CO', type=Meta.DISC, values=range(2, 11), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=4, name="Effort", type=Meta.CONT, range=(460, 26670), is_obj=True)]

  @staticmethod
  def data():
    return read_csv('datasets/csv/finnish.arff.csv')


if __name__ == "__main__":
  print(Finnish().get_rows())
