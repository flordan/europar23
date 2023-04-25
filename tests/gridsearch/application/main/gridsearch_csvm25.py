import numpy as np

from sklearn import clone, datasets
from sklearn.utils import shuffle


import dislib as ds
from dislib.classification import CascadeSVM
from dislib.model_selection import GridSearchCV

from pycompss.api.task import task
from pycompss.api.api import compss_barrier
from pycompss.api.api import compss_wait_on
import time
import sys

def assertSuccessfulFitting(searcher, n_splits, *attributes):
    for att in attributes:
        if not hasattr(searcher, att):
            raise Exception("searcher has no attribute", att)
            sys.exit(1)
    if searcher.n_splits_ != n_splits:
        raise Exception("Expected", n_splits, "splits. Searcher has", searcher.n_splits)
        sys.exit(1)


def launchGridSearch(x, y, n_splits, param_grid):
    print("Launching Preparing CSVM", flush=True)
    csvm = CascadeSVM(check_convergence=True, max_iter=5, random_state=0)
    print("Preparing GridSearch", flush=True)
    searcher = GridSearchCV(csvm, param_grid, cv=n_splits)

    print("Starting search...", flush=True)
    t0 = time.time()
    searcher.fit(x, y)
    compss_barrier()
    t1 = time.time()
    print("Search completed", flush=True)

    assertSuccessfulFitting(searcher, n_splits, 'best_estimator_', 'best_score_', 'best_params_', 'best_index_', 'scorer_')
    return t1-t0

def main(num_gridsearch=5, dataset="IRIS"):
    """Tests GridSearchCV fit()."""
    print("Loading dataset", flush=True)
    if dataset.upper() == "IRIS":
        print("IRIS", flush=True)
        x_np, y_np = datasets.load_iris(return_X_y=True)
        x_np, y_np = shuffle(x_np, y_np, random_state=0)
        x = ds.array(x_np, (60, 4))
        y = ds.array(y_np[:, np.newaxis], (60, 1))
    elif dataset.upper() == "DIGITS":
        print("DIGITS", flush=True)
        x_np, y_np = datasets.load_digits(return_X_y=True)
        x_np, y_np = shuffle(x_np, y_np, random_state=0)
        x = ds.array(x_np, (50, 64))
        y = ds.array(y_np[:, np.newaxis], (50, 1))
    else:
        print("Atrial Fibrillation", flush=True)
        import pickle
        with open(dataset, 'rb') as data:
            x_np = pickle.load(data)
            num_samples = len(x_np)
            num_features = len(x_np[0])
            y_np = pickle.load(data)
        print("Dataset with " + str(num_samples) + " instances of " + str(num_features) + " samples", flush = True)
        num_sample_blocks = 256 
        samples_per_block = int((num_samples + num_sample_blocks - 1) / num_sample_blocks)
        num_feat_blocks = 1 
        features_per_block = int((num_features + num_feat_blocks - 1) / num_feat_blocks)
        print("Trainset divided in blocks of " + str(samples_per_block) + "x" + str(features_per_block), flush = True)

        x = ds.array(x_np, block_size=(samples_per_block, features_per_block))
        y = ds.array(y_np, block_size=(samples_per_block, 1))
        
    print("Dataset with " + str(len(x_np)) + " instances of " + str(len(x_np[0])) + " samples", flush=True)
    time.sleep(10)

    print("Creating hyperparameters grid", flush=True)
    param_grid = {
        'gamma': [0.1, 0.2, 0.3, 0.4, 0.5],
        'c': [0.1, 0.2, 0.3, 0.4, 0.5]
    }

    n_splits = 5

    # warm-up run
    t = launchGridSearch(x, y, n_splits, param_grid) 
    print("Warm-up execution time: " + str(t), flush=True)

    # real runs
    for _ in range(num_gridsearch):
        t = launchGridSearch(x, y, n_splits, param_grid)
        print("GridSearch execution time: " + str(t), flush=True)


@task()
def main_agents(num_gridsearch="5", dataset="IRIS"):
    num_gridsearch = int(num_gridsearch)
    main(num_gridsearch, dataset)


if __name__ == '__main__':
    main()




