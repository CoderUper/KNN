# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:28:42 2020

@author: 金祝光
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

#data
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['labels']=iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

#print(df)

plt.scatter(df[0:50]['sepal length'],df[0:50]['sepal width'],label='0')
plt.scatter(df[50:100]['sepal length'],df[50:100]['sepal width'],label='1')
plt.scatter(df[100:150]['sepal length'],df[100:150]['sepal width'],label='2')
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.show()

data=np.array(df.iloc[:100,[0,1,-1]])
X,y = data[:,:-1],data[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

class KNN:
    def __init__(self,X_train,y_train,n_neighbor=3,p=2):
        self.n=n_neighbor
        self.p=p
        self.X_train=X_train
        self.y_train=y_train
    def predict(self,X):
        knn_list=[]
        for i in range(self.n):
            dist=np.linalg.norm(X-self.X_train[i],ord=self.p)
            knn_list.append((dist,self.y_train[i]))
        
        for i in range(self.n,len(self.X_train)):
            max_index=knn_list.index(max(knn_list,key=lambda x:x[0]))
            dist=np.linalg.norm(X-self.X_train[i],ord=2)
            if dist<knn_list[max_index][0]:
                knn_list[max_index]=(dist,self.y_train[i])
        
        knn=[k[-1] for k in knn_list]
        count_pairs=Counter(knn)
        max_count=sorted(count_pairs.items(),key=lambda x: x[1])[-1][0]
        return max_count

    def score(self,X_test,y_test):
        right_count=0
        for X,y in zip(X_test,y_test):
            label=self.predict(X)
            if label==y:
                right_count+=1
        return right_count/len(X_test)
        
clf=KNN(X_train,y_train)
print(clf.score(X_test,y_test))

test_point=[6.0,3.0]
print("test point:{}".format(clf.predict(test_point)))

plt.scatter(df[0:50]['sepal length'],df[0:50]['sepal width'],label='0')
plt.scatter(df[50:100]['sepal length'],df[50:100]['sepal width'],label='1')
plt.scatter(test_point[0],test_point[1],label='test_point')
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend()

clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train,y_train)
print(clf_sk.score(X_test,y_test))

        
        
        
        