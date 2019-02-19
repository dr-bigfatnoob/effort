from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"


from datasets.dataset import Dataset, Meta, read_pandas_dataframe
from datasets.cleaned import data_to_use


class Kemerer(Dataset):
  def __init__(self):
    Dataset.__init__(self, data=Kemerer.data(),
                     dec_meta=Kemerer.decision_meta(),
                     obj_meta=Kemerer.objective_meta())

  @staticmethod
  def decision_meta():
    return [Meta(index=0, name="Language", type=Meta.DISC, values=range(1, 4), is_obj=False),
            Meta(index=1, name="Hardware", type=Meta.DISC, values=range(1, 7), is_obj=False),
            Meta(index=2, name="AdjFp", type=Meta.CONT, range=(99, 2308), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=3, name="EffortMM", type=Meta.CONT, range=(23, 1108), is_obj=True)]

  @staticmethod
  def data():
    return read_pandas_dataframe(data_to_use.data_kemerer(), read_header=False)

if __name__ == "__main__":
  print(Kemerer().get_rows())
