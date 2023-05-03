import seaborn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from scipy import stats
import scipy.stats

path = '/Users/katya/Desktop/knots/'

crossing_number = 14
label='n'

data = pd.read_csv(path + f'data/{crossing_number}-crossing/' + f'data_{crossing_number}{label}.csv')

np.random.seed(42)
x = data['volumes']
y = np.log(data['determinants']) #kfh_total_ranks
scipy_results = stats.linregress(x, y)
print(scipy_results)
dof = 1.0*len(x) - 2
print("degrees of freedom = ", dof)

print("r-squared:" , scipy_results.rvalue**2)

print('Pearson product-moment correlation coefficients:')
print(np.corrcoef(x, y))


