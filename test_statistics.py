import numpy as np


def ttest(xmat, ymat, paired, num_edges, tail):
    t_stat = np.zeros((num_edges))
    for i in range(num_edges):
        if paired:
            t_stat[i] = ttest_paired_stat_only(xmat[i, :], ymat[i, :], tail)
        else:
            t_stat[i] = ttest2_stat_only(xmat[i, :], ymat[i, :], tail)

    return t_stat


def ttest_perm(d, paired, num_edges, num_sub_x, num_sub_y, tail):
    t_stat = np.zeros((num_edges))
    for i in range(num_edges):
        if paired:
            t_stat[i] = ttest_paired_stat_only(
                d[i, :num_sub_x], d[i, -num_sub_x:], tail)
        else:
            t_stat[i] = ttest2_stat_only(
                d[i, :num_sub_x], d[i, -num_sub_y:], tail)

    return t_stat


def ttest2_stat_only(x, y, tail):
    t = np.mean(x) - np.mean(y)
    n1, n2 = len(x), len(y)
    s = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1)
                 * np.var(y, ddof=1)) / (n1 + n2 - 2))
    denom = s * np.sqrt(1 / n1 + 1 / n2)
    if denom == 0:
        return 0
    if tail == 'both':
        return np.abs(t / denom)
    if tail == 'left':
        return -t / denom
    else:
        return t / denom


def ttest_paired_stat_only(A, B, tail):
    n = len(A - B)
    df = n - 1
    sample_ss = np.sum((A - B)**2) - np.sum(A - B)**2 / n
    unbiased_std = np.sqrt(sample_ss / (n - 1))
    z = np.mean(A - B) / unbiased_std
    t = z * np.sqrt(n)
    if tail == 'both':
        return np.abs(t)
    if tail == 'left':
        return -t
    else:
        return t
