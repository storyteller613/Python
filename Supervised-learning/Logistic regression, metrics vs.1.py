# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:30:03 2017

@author: JG
"""

# Import numpy, pandas, seaborn, Lasso, Matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

######################################
#Logistic regression and the ROC curve
######################################

# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))