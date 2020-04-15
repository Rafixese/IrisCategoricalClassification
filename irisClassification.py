#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:33:41 2020

@author: ganja
"""
#%%########
# Imports #
###########

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

#%%###################
# Data preprocessing #
######################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

dataset = pd.read_csv('iris.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

Y_labelEncoder = LabelEncoder()
Y_train = Y_labelEncoder.fit_transform(Y_train)
Y_test = Y_labelEncoder.transform(Y_test)

Y_train = np.array(Y_train).reshape( (len(Y_train), 1) )
Y_test = np.array(Y_test).reshape( (len(Y_test), 1) )

ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
Y_train = np.array(ct.fit_transform(Y_train))
Y_test = np.array(ct.transform(Y_test))

# Moved to neural network, confusion matrixes don't like this
# Y_train = Y_train[:, 1:]
# Y_test = Y_test[:, 1:]

#%%#####################
# KNeighborsClassifier #
########################

from sklearn.neighbors import KNeighborsClassifier

neighbor_classifier = KNeighborsClassifier(n_neighbors=5)
neighbor_classifier.fit(X_train, Y_train)

y_pred_train = neighbor_classifier.predict(X_train)

cm = confusion_matrix(Y_train.argmax(axis=1), y_pred_train.argmax(axis=1))

print("KNeighborsClassifier accuracy on train set:", (cm[0,0]+cm[1,1]+cm[2,2]) / len(y_pred_train))

y_pred_test = neighbor_classifier.predict(X_test)

cm = confusion_matrix(Y_test.argmax(axis=1), y_pred_test.argmax(axis=1))

print("KNeighborsClassifier accuracy on test set:", (cm[0,0]+cm[1,1]+cm[2,2]) / len(y_pred_test))

# Accuracy on train set: 0.9583333333333334
# Accuracy on test set: 1.0

#%%##########
#    SVM    #
#############

from sklearn.svm import SVC

svc_classifier = SVC()
svc_classifier.fit(X_train, Y_train.argmax(axis=1))

y_pred_train = svc_classifier.predict(X_train)

cm = confusion_matrix(Y_train.argmax(axis=1), y_pred_train)

print("SVC accuracy on train set:", (cm[0,0]+cm[1,1]+cm[2,2]) / len(y_pred_train))

y_pred_test = svc_classifier.predict(X_test)

cm = confusion_matrix(Y_test.argmax(axis=1), y_pred_test)

print("SVC accuracy on test set:", (cm[0,0]+cm[1,1]+cm[2,2]) / len(y_pred_test))

# Accuracy on train set: 0.9583333333333334
# Accuracy on test set: 1.0














































