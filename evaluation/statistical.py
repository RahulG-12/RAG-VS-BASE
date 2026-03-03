import numpy as np
from scipy import stats

def t_test(sample1, sample2):
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    return t_stat, p_value
