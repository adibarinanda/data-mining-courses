# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 17:10:02 2018

@author: adibarinanda
"""

import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss
import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv('dataset.csv')

X = data.drop('Outcome', axis=1) 
y = data['Outcome']

pca = PCA(n_components=2)
X_vis = pca.fit_transform(X)

print('Original dataset shape {}'.format(Counter(y)))
#Original dataset shape Counter({1: 900, 0: 100})

nm = NearMiss()
X_resampled, y_resampled = nm.fit_sample(X, y)
X_res_vis = pca.transform(X_resampled)

print('Resampled dataset shape {}'.format(Counter(y_resampled)))
#Resampled dataset shape Counter({1: 100, 0: 100})

# Two subplots, unpack the axes array immediately
f, (ax1, ax2) = plt.subplots(1, 2)
c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=1)
c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=1)
ax1.set_title('Original set')

ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1], label="Class #0", alpha=0.5)
ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1], label="Class #1", alpha=0.5)
ax2.set_title('Near Miss')

# make nice plotting
for ax in (ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])

plt.figlegend((c0, c1), ('Class #0', 'Class #1'), loc='lower center', ncol=2, labelspacing=0.)
plt.tight_layout(pad=3)
plt.show()