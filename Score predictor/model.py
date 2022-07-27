#!/usr/bin/env python
# coding: utf-8

# ### Final test score prediction

# In[24]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[17]:


# loading the dataC:\Users\user\Downloads
df = pd.read_csv("C:/Users/user/Downloads/mlr032.csv")


# #### Visualizing the data

# In[18]:


sns.pairplot(df, x_vars=['EXAM1', 'EXAM2','EXAM3'], 
             y_vars='FINAL', height=4, aspect=1, kind='scatter')
plt.show()


# In[20]:


#heatmap
sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
plt.show()


# From the above plots, it is clear that the test scores are highly correlated

# ### Model development

# In[21]:


x = df.drop('FINAL', axis=1)
y = df['FINAL']


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=0)




# In[25]:


regressor = LinearRegression()
regressor.fit(x_train.values, y_train)



# In[29]:


#saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))


# In[30]:


model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[89,96,98]]))
