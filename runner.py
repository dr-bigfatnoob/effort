from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from datasets.albrecht import Albrecht
from datasets.china import China
from datasets.desharnais import Desharnais
from datasets.finnish import Finnish
from datasets.isbsg10 import ISBSG10
from datasets.kemerer import Kemerer
from datasets.kitchenhamm import Kitchenhamm
from datasets.maxwell import Maxwell
from datasets.miyazaki import Miyazaki
from utils.lib import *
from utils.validation import *
from methods.peeking import peeking2
from methods.cart import cart
from methods.teak import teak
from methods.knn import knn_1, knn_3
from methods.cogee import cogee
from methods.atlm import atlm
from optimizer.teak_optimize import teak_optimize
from utils.errors import *
from utils import sk
from joblib import Parallel, delayed
from time import time



datasets = [Albrecht, Desharnais, Finnish, Kemerer, Maxwell,
            Miyazaki, China, ISBSG10, Kitchenhamm]
error = msae


def mre_calc(y_predict, y_actual):
  mre = []
  for predict, actual in zip(y_predict, y_actual):
    mre.append(abs(predict - actual) / (actual))
  mmre = np.median(mre)
  if mmre == 0:
    mmre = np.mean(mre)
  return mmre


def sa_calc(y_predict, y_actual):
  ar = 0
  for predict, actual in zip(y_predict, y_actual):
    ar += abs(predict - actual)
  mar = ar / (len(y_predict))
  marr = sum(y_actual) / len(y_actual)
  sa_error = (1 - mar / marr)
  return sa_error


def run(reps=1):
  for dataset_class in datasets:
    dataset = dataset_class()
    model_scores = {"CART": N(),
                    "PEEKING": N(),
                    "TEAK": N(),
                    "KNN1": N(),
                    "KNN3": N(),
                    "ATLM": N(),
                    "COGEE": N(),
                    "O_TEAK": N()
                    }
    for score in model_scores.values():
      score.go = True
    for _ in xrange(reps):
      for test, rest in kfold(dataset.get_rows(), 3, shuffle=True):
        say(".")
        desired_effort = [dataset.effort(row) for row in test]
        all_efforts = [dataset.effort(one) for one in rest]
        model_scores["PEEKING"] += error(desired_effort, peeking2(dataset, test, rest), all_efforts)
        model_scores["CART"] += error(desired_effort, cart(dataset, test, rest), all_efforts)
        model_scores["TEAK"] += error(desired_effort, teak(dataset, test, rest), all_efforts)
        model_scores["KNN1"] += error(desired_effort, knn_1(dataset, test, rest), all_efforts)
        model_scores["KNN3"] += error(desired_effort, knn_3(dataset, test, rest), all_efforts)
        model_scores["ATLM"] += error(desired_effort, atlm(dataset, test, rest), all_efforts)
        model_scores["COGEE"] += error(desired_effort, cogee(dataset, test, rest), all_efforts)
        model_scores["O_TEAK"] += error(desired_effort, teak_optimize(dataset, test, rest), all_efforts)
    sk_data = [[key] + n.cache.all for key, n in model_scores.items()]
    print("\n### %s (%d projects, %d decisions)" %
          (dataset_class.__name__, len(dataset.get_rows()), len(dataset.dec_meta)))
    print("```")
    sk.rdivDemo(sk_data)
    print("```")
    print("")


def run_for_dataset(dataset_class, dataset_id, reps, folds):
  write_file = "results/%s_sa_mre.txt" % dataset_class.__name__
  with open(write_file, "wb") as f:
    dataset = dataset_class()
    print("\n### %s (%d projects, %d decisions)" %
          (dataset_class.__name__, len(dataset.get_rows()), len(dataset.dec_meta)))
    for rep in range(reps):
      fold_id = 0
      for test, rest in kfold(dataset.get_rows(), folds, shuffle=True):
        print("Running for %s, rep = %d, fold = %d" % (dataset_class.__name__, rep + 1, fold_id))
        fold_id += 1
        actual_efforts = [dataset.effort(row) for row in test]
        start = time()
        atlm_efforts = atlm(dataset, test, rest)
        atlm_end = time()
        cart_efforts = cart(dataset, test, rest)
        cart_end = time()
        cogee_efforts = cogee(dataset, test, rest)
        cogee_end = time()
        atlm_mre, atlm_sa = mre_calc(atlm_efforts, actual_efforts), sa_calc(atlm_efforts, actual_efforts)
        cart_mre, cart_sa = mre_calc(cart_efforts, actual_efforts), sa_calc(cart_efforts, actual_efforts)
        cogee_mre, cogee_sa = mre_calc(cogee_efforts, actual_efforts), sa_calc(cogee_efforts, actual_efforts)
        f.write("%d;%d;%f;%f;%f\n" % (dataset_id, 1, atlm_mre, atlm_sa, atlm_end - start))
        f.write("%d;%d;%f;%f;%f\n" % (dataset_id, 2, cart_mre, cart_sa, cart_end - start))
        f.write("%d;%d;%f;%f;%f\n" % (dataset_id, 3, cogee_mre, cogee_sa, cogee_end - start))
  return write_file


def run_patrick(reps, folds, num_cores):
    consolidated_file = "results/patrick_sa_mre.txt"
    dataset_files = Parallel(n_jobs=num_cores)(delayed(run_for_dataset)(dataset_class, dataset_id, reps, folds)
                                               for dataset_id, dataset_class in enumerate(datasets))
    with open(consolidated_file, "wb") as f:
      f.write("dataset;method;SA;MRE;Runtime\n")
      for dataset_file in dataset_files:
        with open(dataset_file) as df:
          for line in df.readlines():
            if len(line) > 0:
              f.write("%s" % line)
        # os.remove(dataset_file)



def sarro_cogee_dataset(dataset_class, error, folds, reps):
  dataset = dataset_class()
  print("\n### %s (%d projects, %d decisions)" %
        (dataset_class.__name__, len(dataset.get_rows()), len(dataset.dec_meta)))
  model_scores = {"CART": N(),
                  "ATLM": N(),
                  "COGEE": N()
                  }
  for score in model_scores.values():
    score.go = True
  for _ in range(reps):
    for test, rest in kfold(dataset.get_rows(), folds, shuffle=True):
      say(".")
      desired_effort = [dataset.effort(row) for row in test]
      all_efforts = [dataset.effort(one) for one in rest]
      model_scores["CART"] += error(desired_effort, cart(dataset, test, rest), all_efforts)
      model_scores["ATLM"] += error(desired_effort, atlm(dataset, test, rest), all_efforts)
      model_scores["COGEE"] += error(desired_effort, cogee(dataset, test, rest), all_efforts)
  sk_data = [[key] + n.cache.all for key, n in model_scores.items()]
  print("```")
  stat = sk.rdivDemo(sk_data)
  print("```")
  print("")
  write_file = "%s/%s.txt" % ("results/sarro", dataset_class.__name__)
  with open(write_file, "wb") as f:
    f.write("\n### %s (%d projects, %d decisions)\n" %
        (dataset_class.__name__, len(dataset.get_rows()), len(dataset.dec_meta)))
    f.write("```\n%s\n```\n\n" % stat)
  return write_file


def sarro_cogee(num_cores, folds=3, reps=10):
  datasets = [China, Desharnais, Finnish, Maxwell, Miyazaki,
              Albrecht, Kemerer, China, ISBSG10, Kitchenhamm]
  # datasets = [Miyazaki, Finnish]
  mkdir("results/sarro")
  error = msa
  dataset_files = Parallel(n_jobs=num_cores)(delayed(sarro_cogee_dataset)(dataset_class, error, folds, reps)
                                             for dataset_id, dataset_class in enumerate(datasets))
  consolidated_file = "results/sarro/sa.md"
  with open(consolidated_file, "wb") as f:
    for dataset_file in dataset_files:
      with open(dataset_file) as df:
        for line in df.readlines():
          f.write(line)


def _sarro():
  reps = 10
  folds = 3
  cores = 16
  sarro_cogee(cores, folds, reps)


def _main():
  reps = 10
  folds = 3
  cores = 16
  run_patrick(reps, folds, cores)
  # run_patrick(1,2,16)


if __name__ == "__main__":
  # _main()
  _sarro()
