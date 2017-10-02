# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 12:33:24 2017

@author: Rebecca
"""


import numpy as np

from skimage.feature import hog
from funLib import *
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

data_pickle = pickle.load(open('./data_pickle.p','rb'))
X_train = data_pickle['X_train'] 
X_test  = data_pickle['X_test']  
y_train = data_pickle['y_train'] 
y_test  = data_pickle['y_test']  
X_train, y_train = shuffle(X_train, y_train)
## Use a linear SVC 
svc = LinearSVC()
#Cs = [0.1,1,10]
#accu=np.zeros((len(Cs),1))
#
#for i in range(len(Cs)):
#    svc.C = Cs[i]
#    svc.fit(X_train, y_train)
#    accu[i] = svc.score(X_test, y_test)
#    

# it turns out all C values produce same accuracy:  0.989302
# it is pretty high ...
svc.C=1.0
svc.fit(X_train, y_train)  
data_pickle['svc']  = svc
pickle.dump( data_pickle, open('./data_pickle.p','wb'))

# for a linear SVC, the following cross validation grid search is not used for now
parameters = {'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10]}
#svc = svm.SVC(kernel='linear')
svc = LinearSVC()
parameters = {'C':[0.1, 1, 10]}
Cs = [0.1, 1, 10]
#t = time.time()
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), n_jobs=-1)
clf.fit(X_train[:1000,:], y_train[:1000])
#t2 = time.time()
#print(round(t2-t, 2), 'Seconds to train SVC...')







