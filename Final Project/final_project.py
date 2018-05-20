# -*- coding: utf-8 -*-
"""
Created on Mon May 21 00:09:43 2018

@author: adibarinanda
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:16:15 2018

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
import pandas as pd
from IPython.display import display

print (__doc__)

data = pd.read_csv('data.csv')

data = data.drop('Unnamed: 32', axis=1)

id_data = data['id']

labels = data['diagnosis']

print (labels.value_counts())


data = data.drop('diagnosis', axis=1)
labels = labels.map({'B': 0, 'M': 1})
display(data.head())
print ("Feature columns ({} total features):\n{}").format(len(data.columns), list(data.columns))