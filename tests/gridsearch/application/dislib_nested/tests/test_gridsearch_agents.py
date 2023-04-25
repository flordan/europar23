import unittest
import numpy as np

from sklearn import clone, datasets
import time

import dislib as ds
from dislib.classification import CascadeSVM, RandomForestClassifier
from dislib.cluster import DBSCAN, KMeans, GaussianMixture
from dislib.decomposition import PCA
from dislib.neighbors import NearestNeighbors
from dislib.preprocessing import StandardScaler
from dislib.recommendation import ALS
from dislib.regression import LinearRegression
from dislib.model_selection import GridSearchCV, KFold
from dislib.utils import shuffle
from pycompss.api.task import task
from pycompss.api.api import compss_barrier
from pycompss.api.api import compss_wait_on
import sys


def assertSuccessfulFitting(searcher, n_splits, *attributes):
    for att in attributes:
        if not hasattr(searcher, att):
            raise Exception("searcher has no attribute", att)
            sys.exit(1)
    if searcher.n_splits_ != n_splits:            
        raise Exception("Expected", n_splits, "splits. Searcher has", searcher.n_splits)
        sys.exit(1)


def gridSearch(dataset, x_blocksize, y_blocksize):
    """Tests GridSearchCV fit() with different data."""
    x_np, y_np = datasets.load_breast_cancer(return_X_y=True)
    n_splits = 4
    n_gridSearch = 5
    print("______x_blocksize:", x_blocksize)
    print("______y_blocksize:", y_blocksize)
    print("______n_splits:", n_splits)
    

#    param_grid = {'c': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    param_grid = {'c': [0.1], 'gamma': [0.1]}
    classifier = CascadeSVM(check_convergence=False, max_iter=2)
    print("______parameters:", param_grid)
    print("______classifier:", type(classifier))
    searcher = GridSearchCV(classifier, param_grid, cv=n_splits)
    sys.stdout.flush()

    t0 = time.time()
    searcher.fit(x_np, y_np, x_blocksize, y_blocksize)
    compss_barrier()
    t1 = time.time()
    assertSuccessfulFitting(searcher, n_splits, 'best_estimator_', 'best_score_', 'best_params_', 'best_index_', 'scorer_')
    print("______Time warmup gridsearch: ", int((t1 - t0) * 1000 ) )


    for _ in range(n_gridSearch):
        classifier = CascadeSVM(check_convergence=False, max_iter=10)
        searcher = GridSearchCV(classifier, param_grid, cv=n_splits)
        sys.stdout.flush()

        t0 = time.time()
        searcher.fit(x_np, y_np, x_blocksize, y_blocksize)
        compss_barrier()
        t1 = time.time()
        assertSuccessfulFitting(searcher, n_splits, 'best_estimator_', 'best_score_', 'best_params_', 'best_index_', 'scorer_')
        print("______Time gridsearch: ", int((t1 - t0) * 1000 ) )

@task()
def main(argv=""):
    print("______ GUSTAVO")
    print("______ executing dataset:", argv)
    if(argv == "iris"):
        gridSearch(datasets.load_iris(return_X_y=True), (50,4), (50,1))
    elif(argv == "breast"):
        gridSearch(datasets.load_breast_cancer(return_X_y=True), (30,10), (30,1))
    elif(argv == "iaspring"):
        gridSearchIASpring()
    else:
      raise Exception("unknown dataset")

if __name__ == '__main__':
    main("iris")

