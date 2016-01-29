# tests for psydata functions.

import psyutils as pu
from nose.tools import *
from pandas.util.testing import assert_frame_equal
import pandas as pd


def test_expand_grid():
    df = pu.psydata.expand_grid({'height': [60, 70],
                                 'weight': [100, 140, 180]})

    desired = pd.DataFrame({'height': [60, 60, 60, 70, 70, 70],
                            'weight': [100, 140, 180, 100, 140, 180]})

    # make sure column indices are sorted:
    df.sort_index(axis=1, inplace=True)
    desired.sort_index(axis=1, inplace=True)
    assert_frame_equal(df, desired)


def test_binomial_binning():
    df = pu.psydata.expand_grid({'x': [0.1, 0.2],
                                 'correct': [0, 0, 1, 1, 1],
                                 'group': ['A', 'B']})

    binned = pu.psydata.binomial_binning(df, grouping_variable='group')

    desired = pd.DataFrame({'ci_max': {0: 0.83251190593629287, 1: 0.83251190593629287},
                            'ci_min': {0: 0.30790471501167721, 1: 0.30790471501167721},
                            'error_max': {0: 0.2491785726029595, 1: 0.2491785726029595},
                            'error_min': {0: 0.27542861832165616, 1: 0.27542861832165616},
                            'group': {0: 'A', 1: 'B'},
                            'n_successes': {0: 7, 1: 7},
                            'n_trials': {0: 12, 1: 12},
                            'prop_corr': {0: 0.58333333333333337, 1: 0.58333333333333337}})

    binned.sort_index(axis=1, inplace=True)
    desired.sort_index(axis=1, inplace=True)
    assert_frame_equal(binned, desired, check_names=False)
