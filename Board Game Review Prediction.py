#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas
import matplotlib
import seaborn
import sklearn


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[3]:


# Load Data
games = pandas.read_csv("games.csv")


# In[4]:


# Names of columns
print(games.columns)
print(games.shape )


# In[5]:


# Histogram of all the ratings
plt.hist(games["average_rating"])
plt.show()


# In[7]:


# Print first row of all games with 0 score
print(games[games["average_rating"] == 0].iloc[0])  # Indexing by position

# Print first row of all games with score greater than 0
print(games[games["average_rating"] > 0].iloc[0])  # Indexing by position


# In[10]:


# Remove rows without user review
games = games[games["users_rated"] > 0]

# Remove rows with missing values
games = games.dropna(axis=0)

# Make a histogram
plt.hist(games["average_rating"])
plt.show()


# In[15]:


# Correlation matrix
corrmat = games.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = 1, square = True)
plt.show()


# In[17]:


# Get all columns from dataframe
columns = games.columns.tolist()

# Filter columns to remove data we don't want
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]

# Store the variable to be predicted
target = "average_rating"


# In[19]:


# Generate training set
train = games.sample(frac = 0.8, random_state = 1)

# Select anything not in training set
test = games.loc[~games.index.isin(train.index)]

# Print shapes
print(train.shape)
print(test.shape)


# In[20]:


# Import linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize the model class
LR = LinearRegression()

# Fit the model on training data
LR.fit(train[columns], train[target])


# In[21]:


# Generate the predictions for test set
predictions = LR.predict(test[columns])

# Computer error between test predictions and actual values
mean_squared_error(predictions, test[target])


# In[22]:


# Import random forest model
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
RFR = RandomForestRegressor(n_estimators = 100, min_samples_leaf=10, random_state = 1)

# Fit the data
RFR.fit(train[columns], train[target])


# In[23]:


# Make predictions
predictions = RFR.predict(test[columns])

# Computer error
mean_squared_error(predictions, test[target])


# In[26]:


test[columns].iloc[500]


# In[27]:


# Make predictions with both samples
rating_LR = LR.predict(test[columns].iloc[500].values.reshape(1, -1))
rating_RFR = RFR.predict(test[columns].iloc[500].values.reshape(1, -1))

# Print rating LR
print(rating_LR)
print(rating_RFR)

# Real rating
print(test[target].iloc[500])

