# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:28:26 2018

@author: adibarinanda
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import timeit

print (__doc__)

timer_log=[]

def KSVM(clean_data, label2):
    start=timeit.default_timer()
    
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
    #split trainset sama testset data KSVM
    fitur_baru_train, fitur_baru_test, label2_train, label2_test = train_test_split(fitur_baru, label2, test_size = 0.41, random_state=42)
    #panggil library svm
    classify_ksvm = svm.SVC()
    #fit record sama label ke fungsi svm
    classify_ksvm.fit(fitur_baru_train, label2_train)
    test_predicted_class_ksvm = classify_ksvm.predict(fitur_baru_test)
    #train_predicted_class = classify.predict(fitur_baru_train)
    #train_accuracy = confusion_matrix(label2_train, train_predicted_class)
    test_accuracy_ksvm = confusion_matrix(label2_test, test_predicted_class_ksvm)
    #hitung accuracy testset
    overall_test_accuracy_ksvm = test_accuracy_ksvm[0,0]/181*100
    
    stop=timeit.default_timer()
    time_result=stop-start
    timer_log.append("Waktu klasifikasi dengan SVM : "+str(time_result)+" detik")
    
    return overall_test_accuracy_ksvm

def SVM(clean_data, label2):
    start=timeit.default_timer()
    
    clean_data_train, clean_data_test, label2_train, label2_test = train_test_split(clean_data, label2, test_size=0.41, random_state=42)
    classify_svm = svm.SVC()
    classify_svm.fit(clean_data_train, label2_train)
    test_predicted_class_svm = classify_svm.predict(clean_data_test)
    test_accuracy_svm = confusion_matrix(label2_test, test_predicted_class_svm)
    overall_test_accuracy_svm = test_accuracy_svm[0,0]/234*100
    
    stop=timeit.default_timer()
    time_result=stop-start
    timer_log.append("Waktu klasifikasi dengan K-SVM : "+str(time_result)+" detik")
    
    return overall_test_accuracy_svm

def main():
    print (__doc__)

    data = pd.read_csv('data.csv')
    data = data.drop('Unnamed: 32', axis=1)
    #Ambil classnya sebelum di drop
    label2 = data['diagnosis']
    #Drop id sama diagnosis
    clean_data = data.drop('id', axis=1)
    clean_data = clean_data.drop('diagnosis', axis=1)
    
    print("Test accuracy with K-SVM = {}".format(KSVM(clean_data, label2)), "%")
    print("Test accuracy with SVM = {}".format(SVM(clean_data, label2)), "%")
    
    for element in timer_log:
        print (element)

if __name__=="__main__":
	main()