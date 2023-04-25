import numbers
import dislib as ds
from dislib.preprocessing import StandardScaler
import numpy as np
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import COLLECTION_IN, Depth, Type
from pycompss.api.constraint import constraint
import sys


@task()
def fit_batch(estimator, x_train, y_train, x_block_size, y_block_size, parameters, fit_params):
    ds_x_train = ds.array(x_train, block_size=x_block_size)
    ds_x_train = StandardScaler().fit_transform(ds_x_train)
    ds_y_train = ds.array(y_train.reshape(-1, 1), block_size=y_block_size)
    if parameters is not None:
        estimator.set_params(**parameters)
    estimator.fit(ds_x_train, ds_y_train, **fit_params)
    return estimator


def score_func(estimator, x_test, y_test, x_block_size, y_block_size, scorer):
    ds_x_test = ds.array(x_test, block_size=x_block_size)
    ds_x_test = StandardScaler().fit_transform(ds_x_test)
    ds_y_test = ds.array(y_test.reshape(-1, 1), block_size=y_block_size)
    test_scores = _score(estimator, ds_x_test, ds_y_test, scorer)

    return [test_scores]

def fit_and_score_and_validate(estimator, train, validation, scorer, parameters, fit_params):
    ds_x_train, ds_y_train = train
    ds_x_test, ds_y_test = validation

    (x_traing_blocks, x_traing_top_left_shape, x_traing_reg_shape, x_traing_shape) = ds_x_train.make_sendable_parameter()
    (y_traing_blocks, y_traing_top_left_shape, y_traing_reg_shape, y_traing_shape) = ds_y_train.make_sendable_parameter()
    (x_test_blocks, x_test_top_left_shape, x_test_reg_shape, x_test_shape) = ds_x_test.make_sendable_parameter()
    (y_test_blocks, y_test_top_left_shape, y_test_reg_shape, y_test_shape) = ds_y_test.make_sendable_parameter()
    print("Invoking _fit_and_score_and_validate", flush=True)
    return _fit_and_score_and_validate_task(estimator,
        x_traing_blocks, x_traing_top_left_shape, x_traing_reg_shape, x_traing_shape,
        y_traing_blocks, y_traing_top_left_shape, y_traing_reg_shape, y_traing_shape,
        x_test_blocks, x_test_top_left_shape, x_test_reg_shape, x_test_shape,
        y_test_blocks, y_test_top_left_shape, y_test_reg_shape, y_test_shape,
        scorer, parameters, fit_params)

@constraint(computing_units=9)
@task(is_distributed=True, x_traing_blocks={Type: COLLECTION_IN, Depth: 2},
      y_traing_blocks={Type: COLLECTION_IN, Depth: 2},
      x_test_blocks={Type: COLLECTION_IN, Depth: 2},
      y_test_blocks={Type: COLLECTION_IN, Depth: 2},)
def _fit_and_score_and_validate_task(estimator,x_traing_blocks, x_traing_top_left_shape, x_traing_reg_shape, x_traing_shape, y_traing_blocks, y_traing_top_left_shape, y_traing_reg_shape, y_traing_shape, x_test_blocks, x_test_top_left_shape, x_test_reg_shape, x_test_shape, y_test_blocks, y_test_top_left_shape, y_test_reg_shape, y_test_shape, scorer, parameters,
            fit_params):
        

    print("______ empezando fit and score and validate task")
    sys.stdout.flush()
    ds_x_train = ds.ds_array_from_sendable_parameter((x_traing_blocks, x_traing_top_left_shape, x_traing_reg_shape, x_traing_shape))
    ds_y_train = ds.ds_array_from_sendable_parameter((y_traing_blocks, y_traing_top_left_shape, y_traing_reg_shape, y_traing_shape))
    ds_x_test = ds.ds_array_from_sendable_parameter((x_test_blocks, x_test_top_left_shape, x_test_reg_shape, x_test_shape))
    ds_y_test = ds.ds_array_from_sendable_parameter((y_test_blocks, y_test_top_left_shape, y_test_reg_shape, y_test_shape))

    #fit
    if parameters is not None:
        estimator.set_params(**parameters)
    estimator.fit(ds_x_train, ds_y_train, **fit_params)


    #score
    scores = _score(estimator, ds_x_test, ds_y_test, scorer)

    #validate
    # scores = compss_wait_on(scores)
    for scorer_name, score in scores.items():
                score = compss_wait_on(score)
                scores[scorer_name] = validate_score(score, scorer_name)

    print("______ finalizado fit and score and validate task")
    return [scores]




def _score(estimator, x, y, scorers):
    """Return a dict of scores"""
    scores = {}

    for name, scorer in scorers.items():
        score = scorer(estimator, x, y)
        score = compss_wait_on(score)
        scores[name] = score
    return scores


def validate_score(score, name):
    if not isinstance(score, numbers.Number) and \
            not (isinstance(score, np.ndarray) and len(score.shape) == 0):
        raise ValueError("scoring must return a number, got %s (%s) "
                         "instead. (scorer=%s)"
                         % (str(score), type(score), name))
    return score


def aggregate_score_dicts(scores):
    """Aggregate the results of each scorer
    Example
    -------
    >>> scores = [{'a': 1, 'b':10}, {'a': 2, 'b':2}, {'a': 3, 'b':3},
    ...           {'a': 10, 'b': 10}]
    >>> aggregate_score_dicts(scores)
    {'a': array([1, 2, 3, 10]),
     'b': array([10, 2, 3, 10])}
    """
    return {key: np.asarray([score[key] for score in scores])
            for key in scores[0]}


def check_scorer(estimator, scorer):
    if scorer is None:
        if hasattr(estimator, 'score'):
            def _passthrough_scorer(estimator, *args, **kwargs):
                """Function that wraps estimator.score"""
                return estimator.score(*args, **kwargs)
            return _passthrough_scorer
        else:
            raise TypeError(
                "If a scorer is None, the estimator passed should have a "
                "'score' method. The estimator %r does not." % estimator)
    elif callable(scorer):
        return scorer
    raise ValueError("Invalid scorer %r" % scorer)
