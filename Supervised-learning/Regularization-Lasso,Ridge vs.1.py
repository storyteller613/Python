# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 10:46:54 2017

@author: Y
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

#########################
#Regularization I: Lasso
#########################

df_columns = ['population', 'fertility', 'HIV', 'CO2', 'BMI_male', 'GDP',
       'BMI_female', 'child_mortality']

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()

#########################
#Regularization II: Ridge
#########################

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)
