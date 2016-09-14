from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from datasets.albrecht import Albrecht
from datasets.china import China
from datasets.desharnais import Desharnais
from datasets.maxwell import Maxwell
from datasets.miyazaki import Miyazaki
from utils.lib import *
from utils.validation import *
from methods.peeking import peeking2
from methods.cart import cart
from methods.teak import teak
from methods.knn import knn_1, knn_3
from utils.errors import *
from utils import sk

datasets = [Albrecht, China, Desharnais, Maxwell, Miyazaki]
# datasets = [Maxwell]
error = msae


def run():
  for dataset_class in datasets:
    dataset = dataset_class()
    model_scores = {"CART": N(),
                    "PEEKING": N(),
                    "TEAK": N(),
                    "KNN1": N(),
                    "KNN3": N()}
    for score in model_scores.values():
      score.go = True
    for test, rest in kfold(dataset.get_rows(), 3):
      say(".")
      desired_effort = [dataset.effort(row) for row in test]
      all_efforts = [dataset.effort(one) for one in rest]
      model_scores["PEEKING"] += error(desired_effort, peeking2(dataset, test, rest), all_efforts)
      model_scores["CART"] += error(desired_effort, cart(dataset, test, rest), all_efforts)
      model_scores["TEAK"] += error(desired_effort, teak(dataset, test, rest), all_efforts)
      model_scores["KNN1"] += error(desired_effort, knn_1(dataset, test, rest), all_efforts)
      model_scores["KNN3"] += error(desired_effort, knn_3(dataset, test, rest), all_efforts)
    sk_data = [[key] + n.cache.all for key, n in model_scores.items()]
    print("\n### %s (%d projects, %d decisions)" %
          (dataset_class.__name__, len(dataset.get_rows()), len(dataset.dec_meta)))
    print("```")
    sk.rdivDemo(sk_data)
    print("```")
    print("")

# run()

from methods.cogee import COGEE
dataset = Albrecht()
cogee = COGEE(dataset, dataset.get_rows())
point = cogee.generate_one()
print(point.evaluate_row(dataset, dataset.get_rows()[0]))
