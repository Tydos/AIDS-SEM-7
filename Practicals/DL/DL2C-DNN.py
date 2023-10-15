#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Flatten,Dropout


# In[4]:


df = pd.read_csv('iris.csv')


# In[5]:


df


# In[6]:


df = df.drop('Id', axis=1)


# In[7]:


df.head()


# In[8]:


X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[12]:


model = Sequential()


# In[13]:


model.add(Dense(4, input_shape=(4,), activation='relu'))
model.add(Dense(3, activation='softmax'))
optimizer = keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)


# In[14]:


optimizer1 = keras.optimizers.SGD(learning_rate=0.1,momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)


# In[15]:


optimizer3 = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer3, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)


# In[16]:


optimizer4 = keras.optimizers.Adam(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=optimizer4, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)


# In[18]:


optimizer5 = keras.optimizers.Adam(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=optimizer5, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10,batch_size=64)


# In[20]:


#AdaGrad Optimizer
optimizer6 = keras.optimizers.Adagrad(learning_rate=0.1)
model.compile(loss='categorical_crossentropy', optimizer=optimizer6, metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)


# In[ ]:




