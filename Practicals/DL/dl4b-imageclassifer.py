#!/usr/bin/env python
# coding: utf-8

# In[15]:


import tensorflow as tf
from tensorflow.keras import datasets,layers,models
from numpy import unique,argmax
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout,MaxPool2D
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


(x_train,y_train),(x_test,y_test) = datasets.cifar10.load_data()
x_train.shape


# In[3]:


x_test.shape


# In[4]:


y_train[:5]


# In[5]:


y_train = y_train.reshape(-1,)
y_train[:5]


# In[6]:


y_test = y_test.reshape(-1,)


# In[7]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[10]:


def plot_sample(x,y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(x[index])
    plt.xlabel(classes[y[index]])


# In[13]:


plot_sample(x_train,y_train,4)


# In[14]:


x_train = x_train/255.0
x_test = x_test/255.0


# In[18]:


#ANN
model = Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(128,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)


# In[22]:


#CNN
model = Sequential()
model.add(Conv2D(filters =32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation ='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)


# In[24]:


y_pred = model.predict(x_test)
y_pred[:5]


# In[26]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[27]:


y_test[:5]


# In[29]:


plot_sample(x_test, y_test,3)


# In[30]:


classes[y_classes[3]]


# In[ ]:




