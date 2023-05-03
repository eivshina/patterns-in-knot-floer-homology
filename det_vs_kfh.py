import seaborn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from scipy import stats

import scipy.stats

path = '/Users/katya/Desktop/knots/'

crossing_number = 15
label='n'
#alternating knots
data = pd.read_csv(path + f'data/{crossing_number}-crossing/' + f'data_{crossing_number}{label}.csv')


x = data['kfh_total_ranks']
y = data['determinants']

print(np.sum(x<y))

