#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 19:23:52 2018

@author: uguess
"""

# Import data analysis libraries
import numpy as np
import pandas as pd

# Import training and test dataframes
orig_train = pd.read_csv('input/train.csv')
orig_test = pd.read_csv('input/test.csv')

# View the data in training and test set
orig_train.head()
orig_test.head()

# Drop unnecessary columns
training_set = orig_train.drop(['PassengerId', 'Ticket', 'Cabin'], axis = 1)
test_set = orig_test.drop(['PassengerId', 'Ticket', 'Cabin'], axis = 1)

# View statistical information about the data
stats_train = training_set.describe(include='all')
stats_test = test_set.describe(include='all')

# Now lets look at the count of all the columns to see if any column has missing values
stats_train.loc['count', :] 
# Training set is missing Age and Embarked data

stats_test.loc['count', :]
# Test set is missing Age and Fare data

# Data Visualization
# Lets look at some correlation between feature and label
import matplotlib.pyplot as plt
import seaborn as sns

# Lets examine relation between Sex, Age and Survived. 
fig = plt.subplots(figsize=(8, 6))
sns.barplot(x=training_set.Sex, y=training_set.Survived)
plt.ylabel('Passengers Survived(%)')
# So we can clearly see that female passengers are more likely to have 
# survived than male passengers

ax = sns.kdeplot(
        training_set.loc[training_set['Survived'] == 0, 'Age'].dropna(), 
        color='red', 
        label='Did not survive')
ax = sns.kdeplot(
        training_set.loc[training_set['Survived'] == 1, 'Age'].dropna(), 
        color='green', 
        label='Survived')
plt.xlabel('Age')
plt.ylabel('Passengers Survived(%)')
# Here we see that passengers between 20 to 40 have almost same
# percentage of survival i.e. around 3

# Lets examine relation between Pclass and Survived
sns.barplot(x=training_set.Pclass, y=training_set.Survived)
plt.ylabel('Passengers Survived(%)')
# We can tell that Passenger at Pclass 1 i.e. Upper class have better
# chances of survival 

# Embarked vs Survived
sns.pointplot(x=training_set.Embarked, y=training_set.Survived)

# Lets analyze distribution of Age, Embarked and Fare data 
# to fill their missing values
# Age - If you look into the data, the Age data is missing 177 values. 
sns.distplot(training_set['Age'].dropna(), rug=True, kde=True)
# Looking at the distribution plot, Age definitely has some outliers
# Also if we take a look at upper 75 percentile
print(stats_train.loc['75%', 'Age'])
print(stats_test.loc['75%', 'Age'])

# So, lets get a mean of all Age data to fill missing values
training_set['Age'] = training_set['Age'].fillna(stats_train.loc['mean', 'Age'])
test_set['Age'] = test_set['Age'].fillna(stats_test.loc['mean', 'Age'])

# Embarked - Since this is categorical data and there are three possible
# values, we will pick the one with highest frequency i.e. mode
from statistics import mode
mode_embarked = mode(training_set['Embarked'])
training_set['Embarked'] = training_set['Embarked'].fillna(mode_embarked)

# Fare - one missing value for test set
# So, lets get the Fare for same pclass with same family members
empty_fare = test_set[test_set['Fare'].isnull()]
# So a passenger with empty fare is a Pclass - 3 passenger with no family, This
# 60 year-old passenger embarked from S
use_fare = test_set[(test_set['Pclass'] == 3) & 
                    (test_set['SibSp'] == 0) & 
                    (test_set['Parch'] == 0) &
                    (test_set['Embarked'] == 'S')]
test_set['Fare'] = test_set['Fare'].fillna(use_fare['Fare'].iloc[0]);

# Check for outliers






















