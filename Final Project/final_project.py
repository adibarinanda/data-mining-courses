# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:16:15 2018

@author: adibarinanda
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

print (__doc__)

data = pd.read_csv('data.csv')

data = data.drop('Unnamed: 32', axis=1)

#Ambil classnya sebelum di drop
col_list = ['diagnosis']
label = data[col_list]
label2 = data['diagnosis']

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

#split trainset sama testset data SVM
clean_data_train, clean_data_test, label2_train, label2_test = train_test_split(clean_data, label2, test_size=0.41, random_state=42)

#split trainset sama testset data KSVM
fitur_baru_train, fitur_baru_test, label2_train, label2_test = train_test_split(fitur_baru, label2, test_size = 0.41, random_state=42)

#Digabung fitur baru sama classnya
kmeans_result = pd.concat([fitur_baru, label], axis=1)

#panggil library svm
classify_ksvm = svm.SVC()
classify_svm = svm.SVC()

#fit record sama label ke fungsi svm
classify_ksvm.fit(fitur_baru_train, label2_train)

classify_svm.fit(clean_data_train, label2_train)

test_predicted_class_ksvm = classify_ksvm.predict(fitur_baru_test)
test_predicted_class_svm = classify_svm.predict(clean_data_test)
#train_predicted_class = classify.predict(fitur_baru_train)

#train_accuracy = confusion_matrix(label2_train, train_predicted_class)
test_accuracy_ksvm = confusion_matrix(label2_test, test_predicted_class_ksvm)
test_accuracy_svm = confusion_matrix(label2_test, test_predicted_class_svm)

#hitung accuracy testset
overall_test_accuracy_ksvm = test_accuracy_ksvm[0,0]/181*100
overall_test_accuracy_svm = test_accuracy_svm[0,0]/234*100

print("test accuracy with SVM = {}". format(overall_test_accuracy_svm),"%")
print("test accuracy with K-SVM = {}". format(overall_test_accuracy_ksvm),"%")