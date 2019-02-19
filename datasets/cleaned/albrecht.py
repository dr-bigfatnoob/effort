from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

__author__ = "bigfatnoob"


from datasets.dataset import Dataset, Meta, read_pandas_dataframe
from datasets.cleaned import data_to_use


class Albrecht(Dataset):
  def __init__(self):
    Dataset.__init__(self, data=Albrecht.data(),
                     dec_meta=Albrecht.decision_meta(),
                     obj_meta=Albrecht.objective_meta())

  @staticmethod
  def decision_meta():
    return [Meta(index=0, name="Input", type=Meta.CONT, range=(7, 193), is_obj=False),
            Meta(index=1, name="Output", type=Meta.CONT, range=(12, 150), is_obj=False),
            Meta(index=2, name="Inquiry", type=Meta.CONT, range=(0, 75), is_obj=False),
            Meta(index=3, name="File", type=Meta.CONT, range=(3, 60), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=4, name="Effort", type=Meta.CONT, range=(0.5, 105.2), is_obj=True)]

  @staticmethod
  def data():
    return read_pandas_dataframe(data_to_use.data_albrecht(), read_header=False)


if __name__ == "__main__":
  print(Albrecht.data())
