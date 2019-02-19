from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"

from datasets.dataset import Dataset, Meta, read_pandas_dataframe
from datasets.cleaned import data_to_use


class Kitchenhamm(Dataset):
  def __init__(self):
    Dataset.__init__(self, data=Kitchenhamm.data(),
                     dec_meta=Kitchenhamm.decision_meta(),
                     obj_meta=Kitchenhamm.objective_meta())

  @staticmethod
  def decision_meta():
    return [Meta(index=0, name='code', type=Meta.DISC, values=range(1, 7), is_obj=False),
            Meta(index=1, name='type', type=Meta.DISC, values=range(0, 7), is_obj=False),
            Meta(index=2, name='function_points', type=Meta.CONT, range=(15.36, 18138.48), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=3, name="Effort", type=Meta.CONT, range=(6.13, 10.2), is_obj=True)]

  @staticmethod
  def data():
    return read_pandas_dataframe(data_to_use.data_kitchenham(), read_header=False)


if __name__ == "__main__":
  print(Kitchenhamm.data())
