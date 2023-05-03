import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from scipy import stats
from matplotlib.ticker import FuncFormatter
import scipy
import warnings
import math
warnings.filterwarnings("ignore")
path = '/Users/katya/Desktop/knots/'

#f(x) density of knots with small total ranks of knot Floer homology

fig = plt.figure(figsize=(15, 10), dpi=75)
fs = 20
sns.set(font_scale=0.5)
fig.subplots_adjust(hspace=0.525, wspace=0.325)
fig_labels = ['a', 'b', 'c', 'd', 'e', 'f']

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)


cut_rank = 50
parameters =[]
parameters_errors=[]
for k in range(1, 7):
    crossing_number = 11 + k

    sns.set_context("talk")
    sns.set_style('whitegrid')

    data_n = pd.read_csv(path + f'data/{crossing_number}-crossing/' + f'data_{crossing_number}n.csv')
    data_n['crossing_number'] = 'Non-alternating'
    ranks_n = data_n['kfh_total_ranks']
    volumes_n = data_n['volumes']

    volume_cuts_n = np.linspace(volumes_n.min(), volumes_n.max(), 30)

    counts_a = []
    counts_n = []

    for i in range(1, volume_cuts_n.shape[0]):
        ranks_n_in_bin = ranks_n[volumes_n < volume_cuts_n[i]].count()
        cut_ranks_n_in_bin = ranks_n[(volumes_n < volume_cuts_n[i]) & (ranks_n < cut_rank)].count()
        if ranks_n_in_bin != 0:
            counts_n.append(cut_ranks_n_in_bin/ranks_n_in_bin)
        else:
            counts_n.append(0)

    counts_data_n = pd.DataFrame(data={'counts': counts_n, 'volume_cuts': volume_cuts_n[1:], \
                                       f'{crossing_number}-crossing knots': \
                                           np.full(len(counts_n), 'Non-alternating')})

    plt.subplot(3, 2, k)
    palette = {'Non-alternating': 'blue', 'Alternating': 'orange'}
    s = sns.scatterplot(data=counts_data_n, x='volume_cuts', y="counts", style=f'{crossing_number}-crossing knots', palette='Blues', legend=False)
    s.set_title(f'{crossing_number} crossings', fontdict={'fontsize': fs})

    if k == 5 or k==6:
        plt.xlabel('Hyperbolic volume', fontsize=fs)
    else:
        plt.xlabel('', fontsize=fs)

    if k == 1 or k == 3 or k == 5:
        plt.ylabel('Fraction', fontsize=fs)
    else:
        plt.ylabel('', fontsize=fs)

    plt.yticks(np.arange(0, 1.25, step=0.25))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: round(x, 3)))
    s.set_yticklabels(s.get_yticks(), size=fs)

    p0 = [max(counts_data_n["counts"]), np.median(counts_data_n["volume_cuts"]), 1, min(counts_data_n["counts"])]

    # #hack
    if k==1:
        params, cv = scipy.optimize.curve_fit(sigmoid, counts_data_n['volume_cuts'][1:], counts_data_n["counts"][1:], p0)
    else:
        params, cv = scipy.optimize.curve_fit(sigmoid, counts_data_n['volume_cuts'], counts_data_n["counts"], p0)

    L, x0, k, b = params
    xs = np.linspace(counts_data_n['volume_cuts'].min(), counts_data_n['volume_cuts'].max(), 100)
    sns.lineplot(x=xs, y=sigmoid(xs, L, x0, k, b), palette=['black'], legend=False)

    print('params: ', params)
    perr = np.sqrt(np.diag(cv))
    print(perr)
    parameters.append(params)
    parameters_errors.append(perr)

index = [12, 13, 14, 15, 16, 17]
df = pd.DataFrame(data=parameters,  index=index)
df.to_csv(path + f'figs/apr12-prime-num/experiment3/apr28/params_{cut_rank}.csv')

df_err = pd.DataFrame(data=parameters_errors,  index=index)
df_err.to_csv(path + f'figs/apr12-prime-num/experiment3/apr28/params_err_{cut_rank}.csv')

argv3 = f'dist_{cut_rank}.png'
plt.savefig(path + 'figs/apr12-prime-num/experiment3/apr28/' + argv3)