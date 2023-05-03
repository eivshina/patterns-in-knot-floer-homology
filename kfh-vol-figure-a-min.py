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

a_min = [0.8518809454904755, 0.8736610260795222, 0.8940161345554175, 0.9130907594097508, 0.9310170048029057, 0.9479123262027063]
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

    plt.subplot(3, 2, k)
    s = sns.scatterplot(data=data, x=data['volumes'], y=np.log(kfh_total_ranks), \
                        hue=f'{crossing_number}-crossing knots', legend=False, s=4)

    s.set_title(f'{crossing_number}-crossings', fontdict={'fontsize': fs})
    plt.xlabel('Volume', fontsize=fs)
    if k == 1 or k == 3 or k==5:
        plt.ylabel(r'$\log$(r)', fontsize=fs)
    else:
        plt.ylabel('', fontsize=fs)
    s.set_yticklabels(s.get_yticks(), size=fs)

    if k == 5 or k == 6:
        plt.xlabel('Volume', fontsize=fs)
    else:
        plt.xlabel('', fontsize=fs)

    a_vals = np.divide(np.log(kfh_total_ranks), data['volumes'])
    a = a_vals.max()
    idx = np.argmax(a_vals)


    x_max = data['volumes'].iloc[idx]
    y_max = np.log(kfh_total_ranks)[idx]

    print(idx, data['volumes'].shape, x_max, y_max, y_max/x_max)

    s.text(-0.195, .98, fig_labels[k-1], transform=s.transAxes, fontsize=26, weight='bold')
    xs = np.linspace(data['volumes'].min(), data['volumes'].max()/2, 500)
    sns.lineplot(x=xs, y=a*xs)
    sns.scatterplot(x=np.array([x_max]), y=np.array([y_max]))
    sns.despine()


argv3 = f'kfh-vol-a-min.png'
plt.savefig(path + 'figs/kfh-vol/apr28/' + argv3)