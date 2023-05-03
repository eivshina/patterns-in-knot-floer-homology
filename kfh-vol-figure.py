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
fig=plt.figure(figsize=(15,10),dpi=75)
#fig = plt.figure()
fs = 20
sns.set(font_scale=0.5)
fig.subplots_adjust(hspace=0.725, wspace=0.325)
fig_labels = ['a', 'b', 'c', 'd', 'e', 'f']
#a_min = [0.2531, 0.2428, 0.2368, 0.2287, 0.2241, 0.2208]
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

    if k==5 or k==6:
        s = 1
    else:
        s = 4

    s = sns.scatterplot(data=data, x=data['volumes'], y=np.log(kfh_total_ranks), hue = 'crossing_number', legend=False, s=s) #hue=f'{crossing_number}-crossing knots'
    # s1 = sns.scatterplot(data=counts_data_n, x='volume_cuts', y="counts")
    #s.set_title(f'{crossing_number}-crossings')
    s.set_title(f'{crossing_number} crossings', fontdict={'fontsize': fs})
    plt.xlabel('Hyperbolic volume', fontsize=fs)
    if k == 1 or k == 3 or k==5:
        plt.ylabel(r'$\log$(r)', fontsize=fs)
    else:
        plt.ylabel('', fontsize=fs)
    s.set_yticklabels(s.get_yticks(), size=fs)

    if k == 5 or k==6:
        plt.xlabel('Hyperbolic volume', fontsize=fs)
    else:
        plt.xlabel('', fontsize=fs)
    #s.yaxis.set_major_locator(ticker.MultipleLocator(2))
    #yticks = np.arange(round(np.log(data['kfh_total_ranks'].min()))-2,round(np.log(data['kfh_total_ranks'].max()))+5,2)
    #s.set_yticks(yticks)
    #s.yaxis.set_major_locator(MaxNLocator(integer=True))
    #plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    #s.text(-0.19, 7, fig_labels[k-1], fontsize=26, weight='bold')
    s.text(-0.195, .98, fig_labels[k-1], transform=s.transAxes, fontsize=26, weight='bold')
    sns.despine()

#argv3 = f'per_volume_bin.pdf'
#plt.savefig(path + 'figs/distributions/' +  argv3)
argv3 = f'kfh-vol.png'
plt.savefig(path + 'figs/kfh-vol/apr28/' + argv3)