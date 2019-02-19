from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from datasets.dataset import Dataset, Meta, read_pandas_dataframe
from datasets.cleaned import data_to_use


class Miyazaki(Dataset):
  def __init__(self):
    Dataset.__init__(self, data=Miyazaki.data(),
                     dec_meta=Miyazaki.decision_meta(),
                     obj_meta=Miyazaki.objective_meta())

  @staticmethod
  def decision_meta():
    return [Meta(index=0, name="SCRN", type=Meta.CONT, range=(0, 150), is_obj=False),
            Meta(index=1, name="FORM", type=Meta.CONT, range=(0, 76), is_obj=False),
            Meta(index=2, name="FILE", type=Meta.CONT, range=(2, 100), is_obj=False),
            Meta(index=3, name="ESCRN", type=Meta.CONT, range=(0, 2113), is_obj=False),
            Meta(index=4, name="EFORM", type=Meta.CONT, range=(0, 1566), is_obj=False),
            Meta(index=5, name="EFILE", type=Meta.CONT, range=(57, 3800), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=6, name="Effort", type=Meta.CONT, range=(5, 340), is_obj=True)]

  @staticmethod
  def data():
    return read_pandas_dataframe(data_to_use.data_miyazaki(), read_header=False)

if __name__ == "__main__":
  print(Miyazaki.data())
