# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 23:13:47 2017

@author: JG
"""

import numpy as np
import pandas as pd

#import Excel file
data=pd.ExcelFile('book_labels.xlsx')
bookl = data.parse(0)
bl =data.parse(0)
bookl.info
bookl.head()
bookl.tail()

NUMERIC_COLUMNS = ['FTE', 'Total']
LABELS = ['Function',
 'Use',
 'Sharing',
 'Reporting',
 'Student_Type',
 'Position_Type',
 'Object_Type',
 'Pre_K',
 'Operating_Status']

####################
#Summarizing the data
####################

# Print the summary statistics
print(bookl.describe())

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Create the histogram
plt.hist(bookl['FTE'].dropna())

# Add title and labels
plt.title('Distribution of %full-time \n employee works')
plt.xlabel('% of full-time')
plt.ylabel('num employees')

# Display the histogram
plt.show()

############################
#Exploring datatypes in pandas
############################

bookl.dtypes.value_counts()

#########################################
#Encode the labels as categorical variables
#########################################

bookl[LABELS].dtypes

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
bookl[LABELS] = bookl[LABELS].apply(categorize_label, axis=0)

bl['Function'].astype('category')
print(bl['Function'].dtypes)

# Print the converted dtypes
print(bl[LABELS].dtypes)

#    Function            category
#    Use                 category
#    Sharing             category
#    Reporting           category
#    Student_Type        category
#    Position_Type       category
#    Object_Type         category
#    Pre_K               category
#    Operating_Status    category

######################
#Counting unique labels
######################

# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Calculate number of unique values for each label: num_unique_labels
num_unique_labels = bl[LABELS].apply(pd.Series.nunique)

# Plot number of unique values for each label
num_unique_labels.plot(kind='bar')

# Label the axes
plt.xlabel('Labels')
plt.ylabel('Number of unique values')

# Display the plot
plt.show()

#############################
#Computing log loss with NumPy
#############################

import numpy as np

actual_labels = np.array([ 1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.])
correct_confident = np.array([ 0.95,  0.95,  0.95,  0.95,  0.95,  0.05,  0.05,  0.05,  0.05,  0.05])
wrong_not_confident = np.array([ 0.35,  0.35,  0.35,  0.35,  0.35,  0.65,  0.65,  0.65,  0.65,  0.65])
wrong_confident = np.array([ 0.05,  0.05,  0.05,  0.05,  0.05,  0.95,  0.95,  0.95,  0.95,  0.95])
correct_not_confident = np.array([ 0.65,  0.65,  0.65,  0.65,  0.65,  0.35,  0.35,  0.35,  0.35,  0.35])

def compute_log_loss(predicted, actual, eps=1e-14):
    predicted = np.clip(predicted, eps, 1- eps)
    loss = -1 * np.mean(actual * np.log(predicted)
                + (1- actual)
                * np.log(1 - predicted))
    return loss

# Compute and print log loss for 1st case
correct_confident = compute_log_loss(correct_confident, actual_labels)
print("Log loss, correct and confident: {}".format(correct_confident)) 

# Compute log loss for 2nd case
correct_not_confident = compute_log_loss(correct_not_confident, actual_labels)
print("Log loss, correct and not confident: {}".format(correct_not_confident, actual_labels)) 

# Compute and print log loss for 3rd case
wrong_not_confident = compute_log_loss(wrong_not_confident, actual_labels)
print("Log loss, wrong and not confident: {}".format(wrong_not_confident, actual_labels)) 

# Compute and print log loss for 4th case
wrong_confident = compute_log_loss(wrong_confident ,actual_labels)
print("Log loss, wrong and confident: {}".format(wrong_confident, actual_labels)) 

# Compute and print log loss for actual labels
actual_labels = compute_log_loss(actual_labels, actual_labels)
print("Log loss, actual labels: {}".format(actual_labels, actual_labels)) 

