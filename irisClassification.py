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
import matplotlib.pyplot as plt

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

#%%########################
# Building Neural Network #
###########################

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

Y_train = Y_train[:, 1:]
Y_test = Y_test[:, 1:]

def build_classifier(n_neur1, n_neur2, n_neur3, optimizer):
    model = Sequential()
    model.add( Dense(n_neur1, activation = 'relu', input_dim = 4) )
    model.add( Dense(n_neur2, activation = 'relu') )
    model.add( Dense(n_neur3, activation = 'relu') )
    model.add( Dense(2, activation = 'softmax') )
    model.compile(optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])
    return model

#%%######################
# Tuning Neural Network #
#########################

# model = KerasClassifier(build_fn = build_classifier)

# parameters = { 
#     'batch_size' : [8, 16, 32],
#     'n_neur1' : [8, 16, 32],
#     'n_neur2' : [4, 8, 16],
#     'n_neur3' : [4, 8, 16],
#     'optimizer' : ['adam', 'rmsprop'],
#     'epochs' : [80]
#     }

# grid_search = GridSearchCV(model, parameters, scoring = 'accuracy', cv = 10)

# # Y_train.argmax(axis=1) because grid search was confused about categorical output
# grid_search = grid_search.fit(X_train, Y_train.argmax(axis=1), use_multiprocessing = True, workers=8)

# best_parameters = grid_search.best_params_
# best_accuracy = grid_search.best_score_

#%%#############################
# Final fitting Neural Network #
################################

model = build_classifier(32,16,8,'rmsprop')

model.summary()

history = model.fit(X_train, Y_train, batch_size = 32, epochs = 80, validation_data = (X_test, Y_test))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label = 'Train Accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'b', label = 'Train Loss')
plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()













































