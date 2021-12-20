# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 17:25:51 2021

@author: Beboo
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#1.	Read the dataset, convert it to DataFrame and display some from it.
dataset = pd.read_csv("Wuzzuf_Jobs.csv")
dataset.head()

#2.	Display structure and summary of the data.
dataset.describe()

#3.	Clean the data (null, duplications)
print(dataset.isnull().sum()) #there is not null values
dataset.drop_duplicates(subset =None, keep = "first", inplace = True)

#4.	Count the jobs for each company and display that in order (What are the most demanding companies for jobs?)
x = dataset["Company"].value_counts()
print("Number of jobs for each company")
print(x)
print("The most demanding company is **Confidential** ")

#5.	Show step 4 in a pie chart
plt.pie(x, labels = x.index)
plt.show()

#6.	Find out what are the most popular job titles.
y = dataset["Title"].value_counts()
print("The most popular job titles")
print(y)

#7.	Show step 6 in bar chart
plt.bar(y.index, y, color ='maroon', width = 0.4)

#8.	Find out the most popular areas?
z = dataset["Location"].value_counts()
print("The most popular areas")
print(z)

#9.	Show step 8 in bar chart
plt.bar(z.index, z, color ='blue', width = 0.4)

#10. Print skills one by one and how many each repeated and order the output to find out the most important skills required?
dataset["Skills"] = dataset["Skills"].str.split(",", n = 1, expand = True)
skills =dataset["Skills"] .value_counts()
print("Skills one by one and how many each repeated")
print(skills)
print("the most important skills required is: **Sales Skills**")
