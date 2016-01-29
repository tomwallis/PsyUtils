# common helpers for dealing with psychophysical data
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import pandas as pd
import itertools as it
from scipy.stats import beta, binom
from scipy.optimize import minimize
import seaborn as sns
import pdb


def expand_grid(data_dict):
    """ A port of R's expand.grid function for use with Pandas dataframes.
    Taken from:
    `http://pandas.pydata.org/pandas-docs/stable/cookbook.html?highlight=expand%20grid`

    Args:
        data_dict:
            a dictionary or ordered dictionary of column names and values.

    Returns:
        A pandas dataframe with all combinations of the values given.


    Examples::
        import psyutils as pu

        print(pu.misc.expand_grid(
            {'height': [60, 70],
             'weight': [100, 140, 180],
             'sex': ['Male', 'Female']})


        from collections import OrderedDict

        entries = OrderedDict([('height', [60, 70]),
                               ('weight', [100, 140, 180]),
                               ('sex', ['Male', 'Female'])])

        print(pu.misc.expand_grid(entries))

    """

    rows = it.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())


def binomial_binning(dat,
                     y='correct',
                     grouping_variables='x',
                     ci=.95,
                     rule_of_succession=False):
    """Bin trials based on grouping variables, returning a new data frame
    with binomial outcome columns (successes, N_trials, plus propotion correct)
    rather than each row being a single trial.
    This data format can significantly speed up model fitting.

    :param dat:
        a pandas dataframe containing the data. Must have
        grouping_variables columns and also a column corresponding to
        bernoulli outcomes (0, 1).
    :param y:
        A string naming the column of the dataframe corresponding to bernoulli
        trial outcome. Defaults to "correct".
    :param grouping_variables:
        a string or list of strings containing the column names
        to group over. Defaults to 'x'.
    :param ci:
        the percentage of the confidence intervals.
    :param rule_of_succession:
        if true, apply a rule-of-succession correction to the data by
        adding 1 success and one failure to the total number of trials.
        This is essentially a prior acknowledging the possibility of both
        successes and failures, and is used to correct for values with
        proportions of 0 or 1 (to e.g. allow estimation of beta errors).
    :returns:
        a new pandas dataframe where each row is a binomial trial.

    Example
    ----------
    res = binomial_binning(dat, ['subj', 'surround', 'scale'])


    """
    grouped = dat.groupby(grouping_variables, as_index=False)
    res = grouped[y].agg({'n_successes': np.sum,
                          'n_trials': np.size})

    if rule_of_succession:
        res.loc[:, 'n_successes'] += 1
        res.loc[:, 'n_trials'] += 2

    # compute some additional values:
    res.loc[:, 'prop_corr'] = res.n_successes / res.n_trials

    # confidence intervals from a beta distribution:
    cis = beta.interval(ci, res.n_successes, (res.n_trials-res.n_successes))
    res.loc[:, 'ci_min'] = cis[0]
    res.loc[:, 'ci_max'] = cis[1]
    res.loc[:, 'error_min'] = np.abs(res['ci_min'].values -
                                     res['prop_corr'].values)
    res.loc[:, 'error_max'] = np.abs(res['ci_max'].values -
                                     res['prop_corr'].values)
    return(res)
