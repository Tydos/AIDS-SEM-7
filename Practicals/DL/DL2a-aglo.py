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


from tensorflow.keras.datasets import mnist
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0


# In[5]:


X_train.shape


# In[6]:


model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))


# In[7]:


#Stochastic Gradient Des


# In[8]:


optimizer = keras.optimizers.SGD(learning_rate=0.1)
model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])


# In[9]:


model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)


# In[10]:


#Stochastic Gradient Descent with Momentum


# In[11]:


optimizer1 = keras.optimizers.SGD(learning_rate=0.1,momentum=0.9)
model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer1,metrics=['accuracy'])


# In[12]:


model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)


# In[13]:


#Nesterov Accelarated Gradient(NAG)


# In[14]:


optimizer3 = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer3,metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)


# In[15]:


#Adam Optimizer


# In[16]:


optimizer4 = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer4,metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)


# In[17]:


#Mini Batch Gradient Descent with Adam optimizer
optimizer5 = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer5,metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10,batch_size=64)


# In[18]:


#AdaGrad Optimizer
optimizer6 = keras.optimizers.Adagrad(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer6,metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10)


# In[ ]:




