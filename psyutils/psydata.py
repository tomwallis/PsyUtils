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


def logistic(x, m, w):
    """ Unscaled logistic function parameterised with threshold
    m and width w. Adapted from Schütt et al, 2016
    """
    denom = 1 + np.exp(-2 * np.log(20-1) * ((x - m) / w))
    return 1 / denom


def weibull(x, m, w):
    """ Unscaled weibull function parameterised with log threshold
    m and width w. Adapted from Schütt et al, 2016.

    Unlike logistic, m must be in natural log units.
    """
    c = np.log(-np.log(.05)) - np.log(-np.log(.95))
    return 1 - np.exp(np.log(.5) * np.exp(c * ((np.log(x) - m) / w)))


def scaled(x, m, w, lam, gam, S):
    """ Function to scale the upper and lower bounds of an unscaled
    [0, 1] psychometric function.

    :param x:  the stimulus level
    :param m:  threshold (x level at half the unscaled value)
    :param w:  width (distance between .05 and .95 of the unscaled function)
    :param lam:  lambda, the lapse rate (1 - lambda is upper asymptote)
    :param gam:  gamma, the lower asymptote.
    :param S:  the unscaled function, taking x, m and w as input.

    :returns:  p, the expected value.
    """
    unscaled = S(x, m, w)
    return gam + (1 - gam - lam) * unscaled


def _parameter_assignment(pars, fixed):
    """Assign parameters from pars vector and fixed dict.

    """
    # parameter assignment.
    if fixed is None:
        if len(pars) != 4:
            raise Exception('no fixed parameters specified but pars is length \
                             {}.'.format(len(pars)))
        m, w, lam, gam = pars
    else:
        if len(pars) + len(fixed) != 4:
            raise Exception('Free and fixed parameters must be four \
                : m, w, lam, gam.')
        if len(pars) == 1:
            m = pars
            w = fixed['w']
            lam = fixed['lam']
            gam = fixed['gam']
        elif len(pars) == 2:
            m, w = pars
            lam = fixed['lam']
            gam = fixed['gam']
        elif len(pars) == 3:
            m, w, lam = pars
            gam = fixed['gam']
        else:
            raise NotImplementedError
    return m, w, lam, gam


def _loss_fun(pars, x, k, n, S, fixed):
    """ A binomial loss function. Returns negative log likelihood
    for k successes in n binomial trials at stimulus level x, fit with
    psychometric fun S.

    :param pars: the vector of parameters to be fit. Order = (m, w, lam, gam).
    :param x:  the stimulus level; if S is weibull should be in log units.
    :param k:  number of successes
    :param n:  number of trials
    :param S:  the unscaled Sigmoid to fit; a function taking
               (x, m, w) as input.
    :param fixed:  dictionary of values for fixed params,
                   e.g. {'lam': 0, 'gam':0.5} for a 2AFC with
                   no lapse rate.

    :returns:  the negative of the summed log likeihoods.

    """
    yhat = psy_pred(pars, x, S, fixed)
    ll = binom.logcdf(k, n, yhat)
    return -ll.sum()


def psy_fit(dat, x, k='n_successes', n='n_trials', function='logistic',
            fixed=None):
    """ Fit a psychometric function to binomial trials in a Pandas
    dataframe. Implements some hacky bounds on parameters, and is only
    useful for dirty eyeballing of data. For actual thresholds you should
    use a more sophisticated package (e.g. psignifit 4).

    :param dat: a pandas dataframe containing the data.
    :param x: string, specifying the column of dat containing the
              stimulus value.
    :param k:  string specifying column containing number of successes.
    :param n:  string specifying column containing number of trials.
    :param function:  string, 'logistic' or 'weibull'
    :param fixed: a dictionary of fixed parameters and their values. E.g.
                  fixed={'gam': 0.5} would fix the lower asymptote to 0.5.
    """

    if function.lower() == 'logistic':
        S = logistic
    elif function.lower() == 'weibull':
        S = weibull
    else:
        raise NotImplementedError

    # unpack numbers:
    x = dat[x].values
    k = dat[k].values
    n = dat[n].values

    if len(x) > 1:
        m_init = x.mean()
        w_init = np.abs(x.max() - x.min()) / 2
    else:
        # x is a scalar
        m_init = x.mean()
        w_init = 2

    # if function.lower() == 'weibull':
    #     m_init = np.log(m_init)

    if fixed is None:
        # all parameters are free.
        inits = [m_init, w_init, 0.05, 0.05]
        bounds = [(None, None), (None, None),
                  (0, 0.25), (0, 0.25)]
    else:
        if len(fixed) == 1:
            # one param fixed (gamma):
            if 'gam' in fixed.keys():
                inits = [m_init, w_init, 0.05]
                bounds = [(None, None), (None, None),
                          (0, 0.25)]
            else:
                raise NotImplementedError
        elif len(fixed) == 2:
            # gamma and lambda fixed:
            if 'gam' in fixed.keys() and 'lam' in fixed.keys():
                inits = [m_init, w_init]
                bounds = [(None, None), (None, None)]
            else:
                raise NotImplementedError
        elif len(fixed) == 3:
            # gamma, lambda and width fixed:
            if 'gam' in fixed.keys() and \
               'lam' in fixed.keys() and \
               'w' in fixed.keys():
                inits = [m_init]
                bounds = [(None, None)]
            else:
                raise NotImplementedError
        else:
            raise Exception("I haven't implemented a different model yet")

    res = minimize(_loss_fun, inits,
                   args=(x, k, n, S, fixed),
                   bounds=bounds,
                   method='L-BFGS-B')

    if res.success is False:
        raise Warning('Optimiser failed to converge!')
    return res


def psy_pred(pars, x, S, fixed=None):
    """Return p values as a function of x, given free
    parameters pars and the fixed parameter dict.
    """
    m, w, lam, gam = _parameter_assignment(pars, fixed)

    if S.__name__ == 'weibull':
        m = np.log(m)

    return scaled(x, m, w, lam, gam, S)


def _fit_cleaner(*args, **kwargs):
    """Clean up the result of psy_fit so that it gets added more nicely
    into a frame.

    args and kwargs are passed straight to psy_fit.
    """
    # pdb.set_trace()
    res = psy_fit(*args, **kwargs)
    pars = res.x

    if 'fixed' in kwargs.keys():
        fixed = kwargs['fixed']
    else:
        fixed = None

    m, w, lam, gam = _parameter_assignment(pars, fixed)
    return pd.Series({'m': m,
                      'w': w,
                      'lam': lam,
                      'gam': gam})


def psy_params(dat, stim_level, correct,
               grouping_variables=None,
               function='logistic',
               fixed=None):
    """Given a bernoulli Pandas dataframe dat, fit a psychometric function
    to the binomial trials (possibly with subsetting in a groupby operation),
    and return the psychometric function parameters as a new series or
    dataframe.

    :param dat:  the Pandas dataframe containing bernoulli trials.
    :param stim_level:  string, the name of the column containing
                        stimulus level.
    :param correct:  string, the name of the column containing bernoulli
               trial (correct / incorrect).
    :param grouping_variables:  a string or list of strings containing columns
                                of dat to group by.
    :param function:  psychometric function to fit; either 'logistic' or
                      'weibull'
    :param fixed: a dict containing parameters to fix,
                  e.g. {'gam': .5} for 2AFC.

    :returns:  a series or dataframe containing threshold m, width w,
               lower asymptote gam and lapse rate lam, for each grouped cell.

    """
    if grouping_variables is None:
        # no group other than the stimulus levels.
        binned = binomial_binning(dat, y=correct, grouping_variables=stim_level)
        res = _fit_cleaner(binned, x=stim_level, function=function, fixed=fixed)
    else:
        # group data into binomial trials, adding the stimulus level to the
        # grouping variables:
        binned = binomial_binning(dat, y=correct,
                                  grouping_variables=grouping_variables + [stim_level])

        res = binned.groupby(grouping_variables).apply(_fit_cleaner,
                                                       x=stim_level,
                                                       function=function,
                                                       fixed=fixed)
        # reset the multiindex:
        res.reset_index(inplace=True)
    return res


def plot_psy_params(dat, stim_level, correct,
                    x, y,
                    grouping_variables=None,
                    function='logistic',
                    fixed=None,
                    **kwargs):
    """ Plot the parameters of fitted psychometric functions. A wrapper
    for psy_params and Seaborn's factorplot.

    :param dat:  the Pandas dataframe containing bernoulli trials.
    :param stim_level:  string, the name of the column containing
                        stimulus level.
    :param correct:  string, the name of the column containing bernoulli
               trial (correct / incorrect).
    :param x:  string, the variable to plot on the x-axis.
               Should be categorical.
    :param y:  string, the parameter to plot on the y-axis (m, w, lam, gam).
    :param grouping_variables:  a string or list of strings containing columns
                                of dat to group by.
    :param function:  psychometric function to fit; either 'logistic' or
                      'weibull'
    :param fixed: a dict containing parameters to fix,
                  e.g. {'gam': .5} for 2AFC.
    :param kwargs:  keyword arguments passed to sns.factorplot.

    :returns:  a series or dataframe containing threshold m, width w,
               lower asymptote gam and lapse rate lam, for each grouped cell.

    """

    # get the fitted parameters for each group:
    params = psy_params(dat, stim_level, correct,
                        grouping_variables=grouping_variables,
                        function=function,
                        fixed=fixed)

    g = sns.factorplot(x, y, data=params, **kwargs)
    return g


def plot_psy(dat, stim_level, correct,
             x, y,
             grouping_variables=None,
             function='logistic',
             fixed=None,
             jitter=True,
             **kwargs):
    """ Fit and plot psychometric functions. A wrapper
    for psy_params and Seaborn's FacetGrid.

    :param dat:  the Pandas dataframe containing bernoulli trials.
    :param stim_level:  string, the name of the column containing
                        stimulus level.
    :param correct:  string, the name of the column containing bernoulli
               trial (correct / incorrect).
    :param x:  string, the variable to plot on the x-axis.
               Should be categorical.
    :param y:  string, the parameter to plot on the y-axis (m, w, lam, gam).
    :param grouping_variables:  a string or list of strings containing columns
                                of dat to group by.
    :param function:  psychometric function to fit; either 'logistic' or
                      'weibull'
    :param fixed: a dict containing parameters to fix,
                  e.g. {'gam': .5} for 2AFC.
    :param jitter:  jitter the x-position of the points (true or false).
    :param kwargs:  keyword arguments passed to FacetGrid. e.g. facet by rows.

    :returns:  a series or dataframe containing threshold m, width w,
               lower asymptote gam and lapse rate lam, for each grouped cell.

    """

    # get the fitted parameters for each group:
    params = psy_params(dat, stim_level, correct,
                        grouping_variables=grouping_variables,
                        function=function,
                        fixed=fixed)

    # plot as a stripplot:
    g = sns.FacetGrid(params, **kwargs)
    g.map(sns.stripplot, x, y, jitter=jitter)
    return g


# def plot_psy(dat, stim_level, correct,
#              x, y,
#              grouping_variables=None,
#              function='logistic',
#              fixed=None,
#              jitter=True,
#              **kwargs):
#     """ Fit and plot psychometric functions. A wrapper
#     for psy_params and Seaborn's FacetGrid.

#     :param dat:  the Pandas dataframe containing bernoulli trials.
#     :param stim_level:  string, the name of the column containing
#                         stimulus level.
#     :param correct:  string, the name of the column containing bernoulli
#                trial (correct / incorrect).
#     :param x:  string, the variable to plot on the x-axis.
#                Should be categorical.
#     :param y:  string, the parameter to plot on the y-axis (m, w, lam, gam).
#     :param grouping_variables:  a string or list of strings containing columns
#                                 of dat to group by.
#     :param function:  psychometric function to fit; either 'logistic' or
#                       'weibull'
#     :param fixed: a dict containing parameters to fix,
#                   e.g. {'gam': .5} for 2AFC.
#     :param jitter:  jitter the x-position of the points (true or false).
#     :param kwargs:  keyword arguments passed to FacetGrid. e.g. facet by rows.

#     :returns:  a series or dataframe containing threshold m, width w,
#                lower asymptote gam and lapse rate lam, for each grouped cell.

#     """

#     # get the fitted parameters for each group:
#     params = psy_params(dat, stim_level, correct,
#                         grouping_variables=grouping_variables,
#                         function=function,
#                         fixed=fixed)

#     # plot as a stripplot:
#     g = sns.FacetGrid(params, **kwargs)
#     g.map(sns.stripplot, x, y, jitter=jitter)
#     return g
