import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from scipy import stats
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
path = '/Users/katya/Desktop/knots/'


sns.set_context("talk")

#sns.axes_style("whitegrid")




fig=plt.figure(figsize=(15,10),dpi=75)
#fig = plt.figure()
fs = 15
sns.set(font_scale=0.5)
fig.subplots_adjust(hspace=0.625, wspace=0.325)
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
    determinants = data['determinants']

    plt.subplot(3, 2, k)
    s = sns.scatterplot(data=data, x=data['volumes'], y=np.log(determinants), \
                        hue=f'{crossing_number}-crossing knots', legend=False, s=4)
    # s1 = sns.scatterplot(data=counts_data_n, x='volume_cuts', y="counts")
    #s.set_title(f'{crossing_number}-crossings')
    s.set_title(f'{crossing_number}-crossings', fontdict={'fontsize': fs})
    plt.xlabel('Volume', fontsize=fs)
    if k == 1 or k == 3 or k==5:
        plt.ylabel(r'$\log$(Determinant)', fontsize=fs)
    else:
        plt.ylabel('', fontsize=fs)
    s.set_yticklabels(s.get_yticks(), size=fs)

    if k == 5 or k==6:
        plt.xlabel('Volume', fontsize=fs)
    else:
        plt.xlabel('', fontsize=fs)

    #plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    #s.text(-0.19, 7, fig_labels[k-1], fontsize=26, weight='bold')
    s.text(-0.155, .98, fig_labels[k-1], transform=s.transAxes, fontsize=26, weight='bold')

    #s.yaxis.set_major_locator(ticker.MultipleLocator(2))
    #yticks = np.arange(round(np.log(data['determinants'].min()))-2, round(np.log(data['determinants'].max()))+5, 2)
    #print(crossing_number, round(np.log(data['determinants'].min()))-2, round(np.log(data['determinants'].max()))+5)
    #plt.yticks(yticks)
    #sns.despine()

#argv3 = f'per_volume_bin.pdf'
#plt.savefig(path + 'figs/distributions/' +  argv3)
argv3 = f'det-vol.png'
plt.savefig(path + 'figs/det-vol/' + argv3)