import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn import decomposition

dataset = pd.read_csv('first.csv', skiprows=27, header=None)

for column in dataset:
    dataset[column] = dataset[column].replace('?', np.NaN)

print dataset

mode=[0]*23;
for i in range(23):
    mode[i]=dataset[i].mode().ix[0]

for i in range(23):
    dataset[i].fillna(mode[i], axis=0, inplace=True)

#from this point ??? 
pca = PCA(n_components=3)
print pca
pca.fit(dataset)

principalDf.plot.bar()

print principalDF

print "end" 

#print(principalDf)
#print(dataset.describe())
#print(df.head(541))
#print(dataset.isnull().sum())
