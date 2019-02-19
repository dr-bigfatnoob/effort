from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from datasets.dataset import Dataset, Meta, read_pandas_dataframe
from datasets.cleaned import data_to_use


class Maxwell(Dataset):
  def __init__(self):
    Dataset.__init__(self, data=Maxwell.data(),
                     dec_meta=Maxwell.decision_meta(),
                     obj_meta=Maxwell.objective_meta())

  @staticmethod
  def decision_meta():
    return [
            Meta(index=0, name="App", type=Meta.DISC, values=range(1, 6), is_obj=False),
            Meta(index=1, name="Har", type=Meta.DISC, values=range(1, 6), is_obj=False),
            Meta(index=2, name="Dba", type=Meta.DISC, values=range(0, 5), is_obj=False),
            Meta(index=3, name="Ifc", type=Meta.DISC, values=range(1, 3), is_obj=False),
            Meta(index=4, name="Source", type=Meta.DISC, values=range(1, 3), is_obj=False),
            Meta(index=5, name="Telonuse", type=Meta.DISC, values=range(0, 2), is_obj=False),
            Meta(index=6, name="Nlan", type=Meta.DISC, values=range(1, 5), is_obj=False),
            Meta(index=7, name="T01", type=Meta.DISC, values=range(1, 6), is_obj=False),
            Meta(index=8, name="T02", type=Meta.DISC, values=range(1, 6), is_obj=False),
            Meta(index=9, name="T03", type=Meta.DISC, values=range(2, 6), is_obj=False),
            Meta(index=10, name="T04", type=Meta.DISC, values=range(2, 6), is_obj=False),
            Meta(index=11, name="T05", type=Meta.DISC, values=range(1, 6), is_obj=False),
            Meta(index=12, name="T06", type=Meta.DISC, values=range(1, 5), is_obj=False),
            Meta(index=13, name="T07", type=Meta.DISC, values=range(1, 6), is_obj=False),
            Meta(index=14, name="T08", type=Meta.DISC, values=range(2, 6), is_obj=False),
            Meta(index=15, name="T09", type=Meta.DISC, values=range(2, 6), is_obj=False),
            Meta(index=16, name="T10", type=Meta.DISC, values=range(2, 6), is_obj=False),
            Meta(index=17, name="T11", type=Meta.DISC, values=range(2, 6), is_obj=False),
            Meta(index=18, name="T12", type=Meta.DISC, values=range(2, 6), is_obj=False),
            Meta(index=19, name="T13", type=Meta.DISC, values=range(1, 6), is_obj=False),
            Meta(index=20, name="T14", type=Meta.DISC, values=range(1, 6), is_obj=False),
            Meta(index=21, name="T15", type=Meta.DISC, values=range(1, 6), is_obj=False),
            Meta(index=22, name="Size", type=Meta.CONT, values=range(48, 3643), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=23, name="Effort", type=Meta.CONT, range=(583, 63694), is_obj=True)]

  @staticmethod
  def data():
    return read_pandas_dataframe(data_to_use.data_maxwell(), read_header=False)


if __name__ == "__main__":
  print(len(Maxwell.data()[0]))
