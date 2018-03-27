# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 17:10:02 2018

@author: adibarinanda
"""

import matplotlib.pyplot as plt
from collections import Counter
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn import neighbors

data = pd.read_csv('dataset.csv')

knn = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto')

X = data.drop('class', axis=1) 
y = data['class']

besar = data.groupby(['class']).size()
besar = list(besar)

koor_x = ['Setelah Balancing dengan Near Miss','Sebelum Balancing dengan Near Miss']
koor_y = besar

besar_smote = data.groupby(['class']).size()
besar_smote = list(besar_smote)

koor_x_smote = ['Sebelum Balancing dengan SMOTE','Setelah Balancing dengan SMOTE']
koor_y_smote = besar_smote

pca = PCA(n_components=2)
X_vis = pca.fit_transform(X)

print('Under-sampling using Near Miss')
print('==============================')
print('Original dataset shape {}'.format(Counter(y)))
#Original dataset shape Counter({1: 900, 0: 100})

xnya = pd.DataFrame(X)
ynya = pd.DataFrame(y)
ynya.columns = ['class']

scores1 = cross_val_score(knn,xnya, ynya['class'], cv=10)
print("Accuracy value knn dengan data Imbalance: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

nm = NearMiss(random_state=42)
X_under_resampled, y_under_resampled = nm.fit_sample(X, y)
X_und_res_vis = pca.transform(X_under_resampled)

y_res = list(y_under_resampled)

valp = y_res.count('Inactive')
valn = y_res.count('Active')

new_y = []
new_y.append(valp)
new_y.append(valn)

print('Resampled dataset shape {}'.format(Counter(y_under_resampled)))
print('================================================================')
#Resampled dataset shape Counter({1: 100, 0: 100})

xnya_under_resampled = pd.DataFrame(X_under_resampled)
ynya_under_resampled = pd.DataFrame(y_under_resampled)
ynya_under_resampled.columns = ['class']

scores2 = cross_val_score(knn,xnya_under_resampled, ynya_under_resampled['class'], cv=10)
print("Accuracy value knn untuk Near Miss: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

print('Over-sampling using SMOTE')
print('=========================')
print('Original dataset shape {}'.format(Counter(y)))

sm = SMOTE(random_state=42)
X_over_resampled, y_over_resampled = sm.fit_sample(X, y)
X_ov_res_vis = pca.transform(X_over_resampled)

y_res_smote = list(y_over_resampled)

valp_smote = y_res_smote.count('Inactive')
valn_smote = y_res_smote.count('Active')

new_y_smote = []
new_y_smote.append(valp_smote)
new_y_smote.append(valn_smote)

print('Resampled dataset shape {}'.format(Counter(y_over_resampled)))
print('================================================================')

xnya1 = pd.DataFrame(X_over_resampled)
ynya1 = pd.DataFrame(y_over_resampled)
ynya1.columns = ['class']

scores3 = cross_val_score(knn,xnya1, ynya1['class'], cv=10)
print("Accuracy value knn untuk SMOTE: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))

plt.figure(1)
plt.bar(koor_x, new_y, label = 'after Near Miss', color='c', width= 0.5, align = 'center')
plt.bar(koor_x, koor_y, label = 'before Near Miss', color='r', width= -0.3, align = 'edge')

plt.figure(2)
plt.bar(koor_x_smote, new_y_smote, label = 'after SMOTE', color='c', width= 0.5, align = 'center')
plt.bar(koor_x_smote, koor_y_smote, label = 'before SMOTE', color='r', width= -0.3, align = 'edge')

# Two subplots, unpack the axes array immediately
#f, (ax1, ax2) = plt.subplots(1, 2)
#c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Inactive", alpha=0.5)
#c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Active", alpha=0.5)
#ax1.set_title('Original set')

#c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Class #0", alpha=0.5)
#c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Class #1", alpha=0.5)
#ax1.set_title('Original set')

#ax2.scatter(X_und_res_vis[y_under_resampled == 0, 0], X_und_res_vis[y_under_resampled == 0, 1], label="Class #0", alpha=0.5)
#ax2.scatter(X_und_res_vis[y_under_resampled == 1, 0], X_und_res_vis[y_under_resampled == 1, 1], label="Class #1", alpha=0.5)
#ax2.set_title('Near Miss')

# make nice plotting
#for ax in (ax1, ax2):
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    ax.get_xaxis().tick_bottom()
#   ax.get_yaxis().tick_left()
#   ax.spines['left'].set_position(('outward', 10))
#    ax.spines['bottom'].set_position(('outward', 10))
#    ax.set_xlim([-200, 200])
#    ax.set_ylim([-200, 200])

#plt.figlegend((c0, c1), ('Inactive', 'Active'), loc='lower center', ncol=2, labelspacing=0.)
#plt.tight_layout(pad=3)
#plt.show()

#f, (ax1, ax2) = plt.subplots(1, 2)
#c0 = ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label="Inactive", alpha=0.5)
#c1 = ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label="Active", alpha=0.5)
#ax1.set_title('Original set')

#ax2.scatter(X_und_res_vis[y_under_resampled == 0, 0], X_und_res_vis[y_under_resampled == 0, 1], label="Class #0", alpha=0.5)
#ax2.scatter(X_und_res_vis[y_under_resampled == 1, 0], X_und_res_vis[y_under_resampled == 1, 1], label="Class #1", alpha=0.5)
#ax2.set_title('SMOTE')

# make nice plotting
#for ax in (ax1, ax2):
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    ax.get_xaxis().tick_bottom()
#    ax.get_yaxis().tick_left()
#    ax.spines['left'].set_position(('outward', 10))
#    ax.spines['bottom'].set_position(('outward', 10))
#    ax.set_xlim([-200, 200])
#    ax.set_ylim([-200, 200])

#plt.figlegend((c0, c1), ('Inactive', 'Active'), loc='lower center', ncol=2, labelspacing=0.)
#plt.tight_layout(pad=3)
#plt.show()