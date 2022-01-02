# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 12:10:54 2022

@author: Beboo
"""
# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset
dataset =pd.read_csv("diabetes.csv") 
dataset.isna().sum() #check for null values
x = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#---------------------------------------------------

#classification algorithm: Decision  Tree
from sklearn.tree import DecisionTreeClassifier

#Taining
outcomeTree = DecisionTreeClassifier(criterion="entropy", max_depth =2)
outcomeTree.fit(x_train, y_train)

#Predicting
y_pred = outcomeTree.predict(x_test)

#Calculate Acuuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('The accuracy of Decision Tree model is {}%'.format((accuracy_score(y_test, y_pred))*100))

#---------------------------------------------

#classification algorithm: Random Forest
from sklearn.ensemble import RandomForestClassifier

#Taining
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting 
y_pred = classifier.predict(x_test)

#Calculate Acuuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('The accuracy of Random Forest model is {}%'.format((accuracy_score(y_test, y_pred))*100))

#----------------------------------------------

#classification algorithm: KNN
from sklearn.neighbors import KNeighborsClassifier

#Taining
k =10
knn =KNeighborsClassifier(k)
knn.fit(x_train ,y_train)

# Predicting 
y_pred = knn.predict(x_test)

#Calculate Acuuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('The accuracy of KNN model is {}%'.format((accuracy_score(y_test, y_pred))*100))

#-----------------------------------------------

#classification algorithm: Logistic Regression
from sklearn.linear_model import LogisticRegression

#Training
LR = LogisticRegression(C=0.2, solver='liblinear')
LR.fit(x_train,y_train)

#Predicting
y_pred = LR.predict(x_test)

#Calculate Acuuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('The accuracy of Logistic Regression model is {}%'.format((accuracy_score(y_test, y_pred))*100))

#---------------------------------------------------------

print("---------- Conclusion ----------\n")
print("** The acuuracy of Random Forest model is the lowest(72.396%), \n** The accuracy of KNN and Logistic Regression are the same and the highest (78.125%)\n** And the accurcy of Decision Tree Model in between (74.48%)")