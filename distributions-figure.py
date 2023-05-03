import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from scipy import stats
path = '/Users/katya/Desktop/knots/'
crossing_number = 16

sns.set_context("talk")
sns.set_style('whitegrid')

labels = ['a', 'n']

data_n = pd.read_csv(path + f'data/{crossing_number}-crossing/nonrandom/' + f'data_{crossing_number}n.csv')
data_n['crossing_number'] = 'Non-alternating'
data_a = pd.read_csv(path + f'data/{crossing_number}-crossing/nonrandom/' + f'data_{crossing_number}a.csv')
data_a['crossing_number'] = 'Alternating'

data = pd.concat([data_n, data_a], ignore_index=True, axis=0)

volumes = data['volumes']
kfh_total_ranks = data['kfh_total_ranks']
khovanov_homology_ranks = data['kfh_total_ranks']  # ! change later
determinants = data['determinants']

fig = plt.figure(figsize=(15, 10), dpi=75)
fig.subplots_adjust(hspace=0.325, wspace=0.325)
fs = 15
# sns.set(font_scale=fs)


ax1 = plt.subplot(221)
sns.histplot(data=data, x=kfh_total_ranks, hue='crossing_number', color='steelblue', alpha=0.5, kde=True,
             stat='probability', element='step', legend=False)
ax1.set_xlabel('Knot-Floer homology total rank', fontsize=fs)
plt.ylabel('Probability', fontsize=fs)

ax2 = plt.subplot(222)
sns.histplot(data=data, x=determinants, hue='crossing_number', alpha=0.5, kde=True,
             stat='probability', element='step', legend=False)
# plt.ylabel('Probability', fontsize=fs)
ax2.set_xlabel('Knot determinant', fontsize=fs)
ax2.set(ylabel='')

ax3 = plt.subplot(223)
sns.histplot(data=data, x=volumes, hue='crossing_number', alpha=0.5, kde=True, stat='probability',
             element='step', legend=False)
plt.ylabel('Probability', fontsize=fs)
ax3.set_xlabel('Hyperbolic volume', fontsize=fs)

ax4 = plt.subplot(224)
sns.histplot(data=data, x=khovanov_homology_ranks, hue='crossing_number', alpha=0.5, kde=True,
             stat='probability', element='step', legend=False)
ax4.set_xlabel('Khovanov homology total rank', fontsize=fs)
# plt.ylabel('Probability', fontsize=fs)
ax4.set(ylabel='')

plt.tight_layout()

# Add labels to plots
for ax, ann in zip([ax1, ax2, ax3, ax4], ['a', 'b', 'c', 'd']):
    ax.text(-0.16, 1, ann, transform=ax.transAxes, fontsize=26, weight='bold')

argv3 = f'dist_{crossing_number}.pdf'
plt.savefig(path + 'figs/distributions/' + argv3)
argv3 = f'dist_{crossing_number}.png'
plt.savefig(path + 'figs/distributions/' + argv3)