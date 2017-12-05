# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:24:14 2017

@author: JG
"""

#########
#imports
#########

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
type(iris)
print(iris.keys())
type(iris.data), type(iris.target)
iris.data.shape
iris.target_names
X = iris.data
y= iris.target
df = pd.DataFrame(X, columns=iris.feature_names)

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

#import CSV file
data = pd.read_csv('C:/Users\Y\Documents\Python Scripts\Congressional_Voting.csv',index_col=0)

df = data.parse(0)
data=pd.ExcelFile(url)
data=pd.ExcelFile('X_new.xlsx')
X_new = data.parse(0)

from urllib.request import urlopen, Request

# Specify the url
url = "https://drive.google.com/open?id=0Bwoo1bdY50XvVWpKLUZXcURNLXM"

# This packages the request
request = Request(url)

# Sends the request and catches the response: response
response = urlopen(request)

# Extract the response: html
html = response.read()

# Print the html
print(html)

# Be polite and close the response!
response.close()

#############
#Numerical EDA
#############

df.head()
df.info()
df.describe()

###########
#Visual EDA
###########

plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

########################
#k-Nearest Neighbors: Fit
########################

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)


# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party',axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X,y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))

##############################
#The digits recognition dataset
##############################

 # Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

#######################################
#Train/Test Split + Fit/Predict/Accuracy
#######################################

# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print(knn.score(X_test, y_test))

X_train = np.array([[ 0.        , -0.33501649, -1.09493684, 1.39616473,
         1.93967098, -0.19600752],
       [ 0.        ,  2.97281114,  1.42951714, -0.63808502,
        -0.5056698 , -0.19600752],
       [ 0.        , -0.33501649,  0.16729015,  0.20951905,
        -0.5056698 , -0.19600752],
       [ 0.        , -0.33501649,  0.58803248, 1.22664392,
        -0.26113572, -0.19600752],
       [ 0.        , -0.33501649,  1.00877481, -0.12952258,
        -0.5056698 , -0.19600752],
       [ 0.        , -0.33501649, -0.25345218, 1.39616473,
        -0.01660165, -0.19600752]])

X_test = np.array([[ 0.        , -0.33501649, -1.09493684, -0.80760583,
        -0.5056698 , -0.19600752],
       [ 0.        , -0.33501649, -1.09493684, 1.22664392,
        -0.5056698 , -0.19600752],
       [ 0.        ,  0.76759272,  1.21914597, -1.14664746,
        -0.5056698 , -0.19600752],
       [ 0.        , -0.33501649,  1.85025947, -0.97712664,
        -0.5056698 , -0.19600752],
       [ 0.        , -0.33501649,  0.37766131, -0.63808502,
        -0.5056698 , -0.19600752],
       [ 0.        , -0.33501649, -0.04308102, 1.22664392,
         0.22793243, -0.19600752]])

y_train = np.array([6, 5, 3, 8, 5, 3])

y_test = np.array([4, 1, 9, 9, 9, 7, 0, 6, 3, 3, 6, 0, 7, 9, 9, 0, 8, 3, 6, 2, 2, 5, 9,
       2, 8, 7, 0, 3, 0, 0, 3, 0, 9, 5, 5, 5, 1, 4, 3, 2, 0, 1, 6, 7, 2, 3,
       7, 2, 7, 5, 3, 7, 6, 0, 1, 9, 0, 3, 2, 9, 7, 7, 8, 3, 6, 8, 5, 7, 8,
       2, 6, 1, 2, 1, 7, 0, 7, 1, 3, 5, 5, 6, 2, 4, 2, 4, 5, 8, 1, 3, 1, 2,
       4, 5, 8, 6, 9, 6, 0, 7, 6, 8, 5, 6, 5, 9, 5, 9, 4, 8, 4, 1, 5, 7, 4,
       4, 8, 3, 0, 6, 4, 7, 8, 7, 0, 6, 9, 8, 7, 0, 0, 3, 8, 5, 9, 9, 7, 8,
       0, 6, 2, 7, 0, 0, 3, 0, 4, 1, 5, 1, 3, 8, 1, 8, 0, 8, 5, 9, 6, 6, 9,
       1, 4, 1, 5, 9, 4, 5, 8, 6, 7, 6, 9, 3, 8, 0, 9, 2, 9, 5, 4, 2, 0, 6,
       3, 7, 4, 6, 6, 5, 0, 6, 6, 4, 3, 0, 0, 3, 5, 0, 8, 9, 5, 4, 8, 1, 0,
       5, 8, 1, 4, 9, 3, 4, 9, 1, 8, 8, 4, 5, 8, 9, 0, 8, 4, 1, 2, 1, 1, 6,
       8, 7, 3, 2, 0, 5, 5, 7, 4, 3, 2, 7, 4, 6, 0, 9, 4, 7, 7, 4, 2, 3, 9,
       3, 2, 3, 8, 9, 3, 5, 3, 6, 6, 3, 2, 4, 0, 3, 9, 9, 5, 9, 2, 0, 2, 5,
       1, 9, 6, 7, 8, 9, 5, 8, 4, 3, 6, 9, 7, 4, 2, 9, 1, 7, 2, 3, 3, 7, 6,
       3, 3, 0, 7, 0, 2, 3, 4, 3, 3, 9, 1, 6, 1, 5, 1, 4, 5, 5, 3, 9, 8, 9,
       4, 8, 2, 0, 2, 9, 3, 3, 2, 9, 9, 9, 9, 8, 9, 1, 5, 6, 5, 2, 0, 2, 1,
       3, 3, 7, 9, 4, 5, 3, 7, 7, 0, 0, 0, 0, 2, 5, 0, 7, 3, 4, 0, 7, 5, 8,
       2, 4, 6, 7, 2, 6, 0, 2, 7, 4, 6, 5, 5, 6, 8, 3, 6, 3, 2, 9, 4, 5, 4,
       5, 0, 3, 2, 2, 8, 9, 4, 5, 5, 1, 7, 9, 1, 0, 1, 2, 2, 3, 4, 0, 8, 8,
       1, 7, 8, 8, 7, 6, 7, 4, 1, 2, 4, 6, 8, 7, 3, 4, 5, 3, 9, 1, 2, 2, 4,
       5, 6, 8, 1, 6, 0, 0, 5, 1, 0, 4, 6, 7, 0, 8, 2, 2, 6, 8, 1, 2, 7, 3,
       4, 7, 5, 9, 8, 6, 9, 0, 6, 1, 1, 5, 6, 7, 1, 8, 6, 4, 4, 7, 5, 0, 1,
       1, 6, 0, 9, 2, 8, 5, 5, 0, 7, 7, 2, 8, 3, 2, 5, 8, 1, 3, 1, 1, 7, 2,
       3, 6, 8, 7, 1, 1, 2, 5, 8, 0, 6, 7, 6, 9, 0, 9, 3, 2, 1, 1, 1, 7, 2,
       9, 1, 0, 4, 2, 5, 0, 5, 7, 3, 4, 4, 9, 4, 9, 2, 8, 6, 1, 2, 4, 7, 9,
       6, 3, 6, 1, 1, 1, 6, 6, 7, 7, 5, 1, 6, 4, 3, 0, 8, 4, 0, 3, 1, 8, 2,
       8, 2, 4, 7, 3, 1, 1, 3, 2, 2, 6, 7, 7, 2, 4, 4, 4, 9, 2, 4, 5, 7, 5,
       4, 4, 6, 5, 4, 8, 6, 4, 8, 0, 0, 3, 3, 9, 8, 0, 6, 5, 1, 8, 1, 9, 3,
       9, 2, 2, 2, 9, 1, 3, 3, 9, 6, 8, 4, 7, 5, 6, 9, 5, 1, 6, 7, 0, 8, 7,
       6, 1, 1, 3, 4, 6, 3, 7, 7, 2, 8, 2, 5, 9, 7, 0, 9, 6, 5, 5, 3, 8, 8,
       7, 9, 2, 1, 0, 3, 7, 4, 8, 8, 2, 6, 8, 1, 9, 1, 4, 4, 5, 6, 3, 4, 1,
       5, 4, 2, 2, 5, 1, 6, 9, 5, 5, 0, 6, 1, 7, 1, 4, 4, 1, 5, 9, 7, 3, 9,
       8, 0, 8, 8, 0, 0])

############################
#Overfitting and underfitting
############################

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn =  KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
