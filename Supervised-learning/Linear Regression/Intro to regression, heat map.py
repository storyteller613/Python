# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 22:31:07 2017

@author: JG
"""

# Import numpy and pandas
import numpy as np
import pandas as pd
import seaborn as sns

#don't have gaminder.csv
# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

#heat map

sns.heatmap(df.corr(), square=True, cmap='RdYlGn')

X_fertility=np.array([[ 2.73],[ 6.43],[ 2.24],[ 1.4 ],[ 1.96],[ 1.41],[ 1.99],
                      [ 1.89],[ 2.38],[ 1.83],[ 1.42],[ 1.82],[ 2.91],[ 5.27],
                      [ 2.51],[ 3.48],[ 2.86],[ 1.9 ],[ 1.43],[ 6.04],
                      [ 6.48],[ 3.05],[ 5.17],[ 1.68],[ 6.81],[ 1.89],
                      [ 2.43],[ 5.05],[ 5.1 ],[ 1.91],[ 4.91],[ 1.43],
                      [ 1.5 ],[ 1.89],[ 3.76],[ 2.73],[ 2.95],[ 2.32],
                      [ 5.31],[ 5.16],[ 1.62],[ 2.74],[ 1.85],[ 1.97],[ 4.28],
                      [ 5.8 ],[ 1.79],[ 1.37],[ 4.19],[ 1.46],[ 4.12],[ 5.34],
                      [ 5.25],[ 2.74],[ 3.5 ],[ 3.27],[ 1.33],[ 2.12],[ 2.64],
                      [ 2.48],[ 1.88],[ 2.  ],[ 2.92],[ 1.39],[ 2.39],[ 1.34],
                      [ 2.51],[ 4.76],[ 1.5 ],[ 1.57],[ 3.34],[ 5.19],[ 1.42],
                      [ 1.63],[ 4.79],[ 5.78],[ 2.05],[ 2.38],[ 6.82],[ 1.38],
                      [ 4.94],[ 1.58],[ 2.35],[ 1.49],[ 2.37],[ 2.44],[ 5.54],
                      [ 2.05],[ 2.9 ],[ 1.77],[ 2.12],[ 2.72],[ 7.59],[ 6.02],
                      [ 1.96],[ 2.89],[ 3.58],[ 2.61],[ 4.07],[ 3.06],[ 2.58],
                      [ 3.26],[ 1.33],[ 1.36],[ 2.2 ],[ 1.34],[ 1.49],[ 5.06],
                      [ 5.11],[ 1.41],[ 5.13],[ 1.28],[ 1.31],[ 1.43],[ 7.06],
                      [ 2.54],[ 1.42],[ 2.32],[ 4.79],[ 2.41],[ 3.7 ],[ 1.92],
                      [ 1.47],[ 3.7 ],[ 5.54],[ 1.48],[ 4.88],[ 1.8 ],[ 2.04],
                      [ 2.15],[ 6.34],[ 1.38],[ 1.87],[ 2.07],[ 2.11],[ 2.46],
                      [ 1.86],[ 5.88],[ 3.85]])

y = np.array([[ 75.3],[ 58.3],[ 75.5],[ 72.5],[ 81.5],[ 80.4],[ 70.6],[ 72.2],
              [ 68.4],[ 75.3],[ 70.1],[ 79.4],[ 70.7],[ 63.2],[ 67.6],[ 70.9],
              [ 61.2],[ 73.9],[ 73.2],[ 59.4],[ 57.4],[ 66.2],[ 56.6],[ 80.7],
              [ 54.8],[ 78.9],[ 75.1],[ 62.6],[ 58.6],[ 79.7],[ 55.9],[ 76.5],
              [ 77.8],[ 78.7],[ 61. ],[ 74. ],[ 70.1],[ 74.1],[ 56.7],[ 60.4],
              [ 74. ],[ 65.7],[ 79.4],[ 81. ],[ 57.5],[ 62.2],[ 72.1],[ 80. ],
              [ 62.7],[ 79.5],[ 70.8],[ 58.3],[ 51.3],[ 63. ],[ 61.7],[ 70.9],
              [ 73.8],[ 82. ],[ 64.4],[ 69.5],[ 76.9],[ 79.4],[ 80.9],[ 81.4],
              [ 75.5],[ 82.6],[ 66.1],[ 61.5],[ 72.3],[ 77.6],[ 45.2],[ 61. ],
              [ 72. ],[ 80.7],[ 63.4],[ 51.4],[ 74.5],[ 78.2],[ 55.8],[ 81.4],
              [ 63.6],[ 72.1],[ 75.7],[ 69.6],[ 63.2],[ 73.3],[ 55. ],[ 60.8],
              [ 68.6],[ 80.3],[ 80.2],[ 75.2],[ 59.7],[ 58. ],[ 80.7],[ 74.6],
              [ 64.1],[ 77.1],[ 58.2],[ 73.6],[ 76.8],[ 69.4],[ 75.3],[ 79.2],
              [ 80.4],[ 73.4],[ 67.6],[ 62.2],[ 64.3],[ 76.4],[ 55.9],[ 80.9],
              [ 74.8],[ 78.5],[ 56.7],[ 55. ],[ 81.1],[ 74.3],[ 67.4],[ 69.1],
              [ 46.1],[ 81.1],[ 81.9],[ 69.5],[ 59.7],[ 74.1],[ 60. ],[ 71.3],
              [ 76.5],[ 75.1],[ 57.2],[ 68.2],[ 79.5],[ 78.2],[ 76. ],[ 68.7],
              [ 75.4],[ 52. ],[ 49. ]])

# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()

X = np.array([[  3.48110590e+07,   2.73000000e+00,   1.00000000e-01,
          1.23140000e+04,   1.29904900e+02,   2.95000000e+01],
       [  1.98422510e+07,   6.43000000e+00,   2.00000000e+00, 
          7.10300000e+03,   1.30124700e+02,   1.92000000e+02],
       [  4.03818600e+07,   2.24000000e+00,   5.00000000e-01, 
          1.46460000e+04,   1.18891500e+02,   1.54000000e+01]
       [  8.65893420e+07,   1.86000000e+00,   4.00000000e-01, 
          4.08500000e+03,   1.21936700e+02,   2.62000000e+01],
       [  1.31145790e+07,   5.88000000e+00,   1.36000000e+01, 
          3.03900000e+03,   1.32449300e+02,   9.49000000e+01],
       [  1.34954620e+07,   3.85000000e+00,   1.51000000e+01, 
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

################################
#Train/test split for regression
################################

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))

#5-fold cross-validation

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg,X,y,cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

#K-Fold CV comparison

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg,X,y,cv=3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg,X,y,cv=10)
print(np.mean(cvscores_10))

%timeit cross_val_score(reg, X, y, cv = ____)