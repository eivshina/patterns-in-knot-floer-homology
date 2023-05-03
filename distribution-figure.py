import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from scipy import stats
from matplotlib.ticker import FuncFormatter
from scipy.stats import shapiro
path = '/Users/katya/Desktop/knots/'

sns.set_context("talk")

fig=plt.figure(figsize=(15,10),dpi=75)
fs = 15
sns.set(font_scale=0.5)
fig.subplots_adjust(hspace=0.625, wspace=0.325)
fig_labels = ['a', 'b', 'c', 'd', 'e', 'f']
param = 'determinants' #'volumes' #'kfh_total_ranks'
title = 'Knot determinant' #'Hyperbolic volume' #'Total rank of knot Floer homology' #
for k in range(1, 7):
    crossing_number = 11 + k

    sns.set_context("talk")
    sns.axes_style("whitegrid")


    labels = ['a', 'n']

    data_n = pd.read_csv(path + f'data/{crossing_number}-crossing/' + f'data_{crossing_number}n.csv')
    data_n['crossing_number'] = 'Non-alternating'

    data_a = pd.read_csv(path + f'data/{crossing_number}-crossing/' + f'data_{crossing_number}a.csv')
    data_a['crossing_number'] = 'Alternating'

    data = pd.concat([data_n, data_a], ignore_index=True, axis=0)
    kfh_total_ranks = data[param]
    #print('Shapiro result', shapiro(data_a[param]))

    plt.subplot(3, 2, k)

    s = sns.histplot(data=data, x=kfh_total_ranks, hue='crossing_number', alpha=0.5, kde=True,
                 stat='probability', element='step', legend=False)

    s.set(title=f'{crossing_number} crossings')

    if k == 1 or k == 3 or k==5:
        plt.ylabel('Probability', fontsize=fs)
    else:
        plt.ylabel('', fontsize=fs)
    s.set_yticklabels(s.get_yticks(), size=fs)

    if k == 5 or k==6:
        plt.xlabel(title, fontsize=fs)
    else:
        plt.xlabel('', fontsize=fs)

    #plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    #s.text(-0.19, 7, fig_labels[k-1], fontsize=26, weight='bold')
    s.text(-0.155, .98, fig_labels[k-1], transform=s.transAxes, fontsize=26, weight='bold')
    sns.despine()

#argv3 = f'per_volume_bin.pdf'
#plt.savefig(path + 'figs/distributions/' +  argv3)
argv3 = f'{param}_dist.png'
plt.savefig(path + f'figs/distributions/apr28/' + argv3)