from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from datasets.dataset import Dataset, Meta, read_pandas_dataframe
from datasets.cleaned import data_to_use


class China(Dataset):
  def __init__(self):
    Dataset.__init__(self, data=China.data(),
                     dec_meta=China.decision_meta(),
                     obj_meta=China.objective_meta())

  @staticmethod
  def decision_meta():
    return [Meta(index=0, name="Input", type=Meta.CONT, range=(0, 9404), is_obj=False),
            Meta(index=1, name="Output", type=Meta.CONT, range=(0, 2455), is_obj=False),
            Meta(index=2, name="Enquiry", type=Meta.CONT, range=(0, 952), is_obj=False),
            Meta(index=3, name="File", type=Meta.CONT, range=(0, 2955), is_obj=False),
            Meta(index=4, name="Interface", type=Meta.CONT, range=(0, 1572), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=5, name="Effort", type=Meta.CONT, range=(26, 54620), is_obj=True)]

  @staticmethod
  def data():
    return read_pandas_dataframe(data_to_use.data_china(), read_header=False)


if __name__ == "__main__":
  print(China.data())
