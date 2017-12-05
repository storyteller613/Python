# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:37:39 2017

@author: JG
"""

# Import numpy, pandas, seaborn, Lasso, Matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

X = np.array([[  3.48110590e+07,   2.73000000e+00,   1.00000000e-01, ...,
          1.23140000e+04,   1.29904900e+02,   2.95000000e+01],
       [  1.98422510e+07,   6.43000000e+00,   2.00000000e+00, ...,
          7.10300000e+03,   1.30124700e+02,   1.92000000e+02],
       [  4.03818600e+07,   2.24000000e+00,   5.00000000e-01, ...,
          1.46460000e+04,   1.18891500e+02,   1.54000000e+01],
       ..., 
       [  8.65893420e+07,   1.86000000e+00,   4.00000000e-01, ...,
          4.08500000e+03,   1.21936700e+02,   2.62000000e+01],
       [  1.31145790e+07,   5.88000000e+00,   1.36000000e+01, ...,
          3.03900000e+03,   1.32449300e+02,   9.49000000e+01],
       [  1.34954620e+07,   3.85000000e+00,   1.51000000e+01, ...,
          1.28600000e+03,   1.31974500e+02,   9.83000000e+01]])

y = np.array([ 75.3,  58.3,  75.5,  72.5,  81.5,  80.4,  70.6,  72.2,  68.4,
        75.3,  70.1,  79.4,  70.7,  63.2,  67.6,  70.9,  61.2,  73.9,
        73.2,  59.4,  57.4,  66.2,  56.6,  80.7,  54.8,  78.9,  75.1,
        62.6,  58.6,  79.7,  55.9,  76.5,  77.8,  78.7,  61. ,  74. ,
        70.1,  74.1,  56.7,  60.4,  74. ,  65.7,  79.4,  81. ,  57.5,
        62.2,  72.1,  80. ,  62.7,  79.5,  70.8,  58.3,  51.3,  63. ,
        61.7,  70.9,  73.8,  82. ,  64.4,  69.5,  76.9,  79.4,  80.9,
        81.4,  75.5,  82.6,  66.1,  61.5,  72.3,  77.6,  45.2,  61. ,
        72. ,  80.7,  63.4,  51.4,  74.5,  78.2,  55.8,  81.4,  63.6,
        72.1,  75.7,  69.6,  63.2,  73.3,  55. ,  60.8,  68.6,  80.3,
        80.2,  75.2,  59.7,  58. ,  80.7,  74.6,  64.1,  77.1,  58.2,
        73.6,  76.8,  69.4,  75.3,  79.2,  80.4,  73.4,  67.6,  62.2,
        64.3,  76.4,  55.9,  80.9,  74.8,  78.5,  56.7,  55. ,  81.1,
        74.3,  67.4,  69.1,  46.1,  81.1,  81.9,  69.5,  59.7,  74.1,
        60. ,  71.3,  76.5,  75.1,  57.2,  68.2,  79.5,  78.2,  76. ,
        68.7,  75.4,  52. ,  49. ])

#import Excel file
data=pd.ExcelFile('Diabetes.xlsx')
X = data.parse(0)
X.head()


y = np.array([1,	0,	1,	0,	1,	0,	1,	0,	1,	1,	0,	1,	0,	1,	1,	1,	1,	1,	0,	1,	0,	0,	
             1,	1,	1,	1,	1,	0,	0,	0,	0,	1,	1,	0,	0,	1,	0,	0,	1,	0,	1,	1,	1,	0,	
             0,	1,	1,	1,	0,	1,	0,	1,	0,	1,	0,	0,	0,	0,	1,	0])	


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

######################################
#Building a logistic regression model
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

######################
#Plotting an ROC curve
######################

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

###############
#AUC computation
###############

# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


#######################################
#Hyperparameter tuning with GridSearchCV
#######################################

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))

#############################################
#Hyperparameter tuning with RandomizedSearchCV
#############################################

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
    
#####################################################################
#Hold-out set in practice I: Classification:  penalties:  'C', L1, L2
#####################################################################

# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

####################################################################
#Hold-out set in practice II: Regression:  elastic net regularization,
#   penalties L1,L2: a∗L1+b∗L2
####################################################################    

# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))



