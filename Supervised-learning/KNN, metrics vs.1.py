# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:28:21 2017

@author: JG
"""

# Import numpy, pandas, seaborn, Lasso, Matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


###########################
#Metrics for classification
###########################

# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))