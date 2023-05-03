import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from scipy import stats
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
path = '/Users/katya/Desktop/knots/'


sns.set_context("talk")

#sns.axes_style("whitegrid")




fig=plt.figure(figsize=(15,10),dpi=75)
#fig = plt.figure()
fs = 20
sns.set(font_scale=0.5)
fig.subplots_adjust(hspace=0.725, wspace=0.325)
fig_labels = ['a', 'b', 'c', 'd', 'e', 'f']

for k in range(1, 7):
    crossing_number = 11 + k

    sns.set_context("talk")
    sns.set_style('darkgrid')


    labels = ['a', 'n']

    data_n = pd.read_csv(path + f'data/{crossing_number}-crossing/' + f'data_{crossing_number}n.csv')
    data_n['crossing_number'] = 'Non-alternating'

    data_a = pd.read_csv(path + f'data/{crossing_number}-crossing/' + f'data_{crossing_number}a.csv')
    data_a['crossing_number'] = 'Alternating'

    data = pd.concat([data_n, data_a], ignore_index=True, axis=0)
    kfh_total_ranks = data['kfh_total_ranks']

    a_vals = np.divide(np.log(kfh_total_ranks), data['volumes'])
    a = a_vals.max()
    idx = np.argmax(a_vals)
    print(crossing_number, a, idx)

