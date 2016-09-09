from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from sklearn.cross_validation import KFold
import numpy as np


def loo(dataset):
  """
  Leave one out experiment
  :param dataset: Dataset object
  :return:
  """
  for index, item in enumerate(dataset):
    yield [item], dataset[:index] + dataset[index + 1:]


def test_train(dataset, test_size=1):
  """
  Split into training and test set
  :param dataset: Dataset object
  :param test_size: Test set size
  :return:
  """
  data = dataset[:]
  dataset_size = len(data)
  if isinstance(test_size, float):
    test_size = int(round(data*test_size))
  for i in range(dataset_size-test_size):
    yield dataset[i:i+test_size], data[:i] + data[i+1:]


def kfold(dataset, n_folds, shuffle=False, random_state=1):
  """
  KFold cross validation technique
  :param dataset:
  :param n_folds:
  :param shuffle:
  :param random_state:
  :return:
  """
  dataset_np = np.array(dataset)
  size = len(dataset)
  kf = KFold(size, n_folds, shuffle=shuffle, random_state=random_state)
  for train_index, test_index in kf:
    yield dataset_np[test_index].tolist(), dataset_np[train_index].tolist()
