# tests for psydata functions.

import psyutils as pu
from pandas.util.testing import assert_frame_equal
import pandas as pd


def test_expand_grid():
    df = pu.psydata.expand_grid({'height': [60, 70],
                                 'weight': [100, 140, 180]})

    desired = pd.DataFrame({'height': [60, 60, 60, 70, 70, 70],
                            'weight': [100, 140, 180, 100, 140, 180]})

    # Because dicts don't follow a consistent order (specified at runtime to
    # reduce memory requirements), we need to sort these and their indices
    # to ensure the assert will pass consistently.

    # same column order
    df.sort_index(axis=1, inplace=True)
    desired.sort_index(axis=1, inplace=True)

    # same row order:
    df.sort_values(by='height', inplace=True)
    desired.sort_values(by='height', inplace=True)

    df.reset_index(drop=True, inplace=True)
    desired.reset_index(drop=True, inplace=True)

    # test
    assert_frame_equal(df, desired, check_names=False)


def test_binomial_binning():
    df = pu.psydata.expand_grid({'x': [0.1, 0.2],
                                 'correct': [0, 0, 1, 1, 1],
                                 'group': ['A', 'B']})

    binned = pu.psydata.binomial_binning(df, grouping_variables='group')

    desired = pd.DataFrame({'ci_max': {0: 0.86300433773483354, 1: 0.86300433773483354},
                            'ci_min': {0: 0.29929505620854041, 1: 0.29929505620854041},
                            'error_max': {0: 0.26300433773483356, 1: 0.26300433773483356},
                            'error_min': {0: 0.30070494379145957, 1: 0.30070494379145957},
                            'group': {0: 'A', 1: 'B'},
                            'n_successes': {0: 6, 1: 6},
                            'n_trials': {0: 10, 1: 10},
                            'prop_corr': {0: 0.59999999999999998, 1: 0.59999999999999998}})

    binned.sort_index(axis=1, inplace=True)
    desired.sort_index(axis=1, inplace=True)
    assert_frame_equal(binned, desired, check_names=False)
