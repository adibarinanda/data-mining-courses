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

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

print (__doc__)

data = pd.read_csv('data.csv')

data = data.drop('Unnamed: 32', axis=1)

#id_data = data['id']
#labels = data['diagnosis']

#Ambil classnya sebelum di drop
col_list = ['diagnosis']
label = data[col_list]

#Drop id sama diagnosis
clean_data = data.drop('id', axis=1)
clean_data = clean_data.drop('diagnosis', axis=1)

#Balik tabelnya buat dikmeans
clean_data_transposed = clean_data.transpose()

#Kmeans pake k=7
kmeans = KMeans(n_clusters=7)
kmeans.fit(clean_data_transposed)

#Ambil center2 barunya
centernya = kmeans.cluster_centers_

#Balik lagi tabelnya
fitur_baru = centernya.transpose()

#Convert jadi DataFrame biar bisa digabung sama classnya
fitur_baru = pd.DataFrame(fitur_baru)

#a = pd.Series(labels).values
#b = np.array(fitur_baru)

#Digabung fitur baru sama classnya
kmeans_result = pd.concat([fitur_baru, label], axis=1)


#data = data.drop('diagnosis', axis=1)
#labels = labels.map({'B': 0, 'M': 1})
#display(data.head())
#print ("Feature columns ({} total features):\n{}").format(len(data.columns), list(data.columns))