from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from dataset import Dataset, Meta, read_csv


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
            Meta(index=4, name="Interface", type=Meta.CONT, range=(0, 1572), is_obj=False),
            Meta(index=5, name="Duration", type=Meta.CONT, range=(1, 84), is_obj=False)]

  @staticmethod
  def objective_meta():
    return [Meta(index=6, name="Effort", type=Meta.CONT, range=(26, 54620), is_obj=False)]

  @staticmethod
  def data():
    return read_csv("datasets/csv/china.arff.csv")
