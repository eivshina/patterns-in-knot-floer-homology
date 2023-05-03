import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from scipy import stats
from matplotlib.ticker import FuncFormatter

path = '/Users/katya/Desktop/knots/'

fig = plt.figure(figsize=(15, 10), dpi=75)
fs = 15
sns.set(font_scale=0.5)
fig.subplots_adjust(hspace=0.325, wspace=0.325)
fig_labels = ['a', 'b', 'c', 'd', 'e', 'f']

for k in range(1, 7):
    crossing_number = 11 + k

    sns.set_context("talk")
    sns.set_style('whitegrid')

    labels = ['a', 'n']

    data_n = pd.read_csv(path + f'data/{crossing_number}-crossing/' + f'data_{crossing_number}n.csv')
    data_n['crossing_number'] = 'Non-alternating'
    ranks_n = data_n['kfh_total_ranks']
    volumes_n = data_n['volumes']

    data_a = pd.read_csv(path + f'data/{crossing_number}-crossing/' + f'data_{crossing_number}a.csv')
    data_a['crossing_number'] = 'Alternating'
    ranks_a = data_a['kfh_total_ranks']
    volumes_a = data_a['volumes']

    data = pd.concat([data_n, data_a], ignore_index=True, axis=0)

    volumes = data['volumes']
    ranks = data['kfh_total_ranks']
    mean_rank = np.mean(ranks)
    std_rank = np.std(ranks)

    cut_rank_a = ranks_a.mean()*0.3
    cut_rank_n = ranks_n.mean()*0.3

    ranks_a = ranks_a[ranks_a < cut_rank_a]
    ranks_n = ranks_n[ranks_n < cut_rank_n]

    miv_volume_cut = volumes.min()  # volumes.mean()+1.2*volumes.std()
    volume_cuts = np.linspace(miv_volume_cut, volumes.max(), 35)

    counts_a = []
    counts_n = []
    volume_cuts_centers = []
    for i in range(1, volume_cuts.shape[0]):
        cut_ranks_count_a = ranks_a[(volumes_a < volume_cuts[i]) & (volumes_a > volume_cuts[i - 1])].count()
        counts_a.append(int(cut_ranks_count_a))
        cut_ranks_count_n = ranks_n[(volumes_n < volume_cuts[i]) & (volumes_n > volume_cuts[i - 1])].count()
        counts_n.append(int(cut_ranks_count_n))
        volume_cut_center = (volume_cuts[i] + volume_cuts[i - 1]) / 2
        volume_cuts_centers.append(volume_cut_center)

    # print('mean volume: ', volumes.mean())
    counts_data_a = pd.DataFrame(data={'counts': counts_a, 'volume_cuts': volume_cuts_centers, \
                                       f'{crossing_number}-crossing knots': np.full(len(counts_a), 'Alternating')})
    counts_data_n = pd.DataFrame(data={'counts': counts_n, 'volume_cuts': volume_cuts_centers, \
                                       f'{crossing_number}-crossing knots': \
                                           np.full(len(counts_n), 'Non-alternating')})

    counts_data = pd.concat([counts_data_a, counts_data_n], ignore_index=True, axis=0)
    plt.subplot(3, 2, k)
    palette = {'Non-alternating': 'blue', 'Alternating': 'orange'}
    s = sns.scatterplot(data=counts_data, x='volume_cuts', y="counts", style=f'{crossing_number}-crossing knots', \
                        hue=f'{crossing_number}-crossing knots', size=f'{crossing_number}-crossing knots', legend=False, palette=palette)
    # = sns.scatterplot(data=counts_data_a, x='volume_cuts', y="counts", style=f'{crossing_number}-crossing knots', legend=False)
    s.set(title=f'{crossing_number}-crossings')

    if k == 5 or k==6:
        plt.xlabel('Volume', fontsize=fs)
    else:
        plt.xlabel('', fontsize=fs)

    if k == 1 or k == 3 or k == 5:
        plt.ylabel('Count', fontsize=fs)
    else:
        plt.ylabel('', fontsize=fs)
    s.set_yticklabels(s.get_yticks(), size=fs)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    # s.text(-0.16, 3, fig_labels[k-1], fontsize=26, weight='bold')



argv3 = f'per_volume_bin_<0p3_mean.png'
plt.savefig(path + 'figs/per-bin/' + argv3)