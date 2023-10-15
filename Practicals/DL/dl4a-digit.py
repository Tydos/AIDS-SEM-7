#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from numpy import unique,argmax
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout,MaxPool2D
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


# In[ ]:





# In[2]:


(x_train,y_train),(x_test,y_test) = mnist.load_data()


# In[4]:


print(x_train.shape)
print(y_train.shape)


# In[5]:


x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],x_test.shape[2],1))


# In[7]:


print(x_train.shape)
print(x_test.shape)


# In[8]:


x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0


# In[14]:


fig = plt.figure(figsize=(10,3))
for i in range(20):
    ax = fig.add_subplot(2,10,i+1,xticks=[],yticks=[])
    ax.imshow(np.squeeze(x_train[i]),cmap='gray')


# In[40]:


shape = x_train.shape[1:]
shape


# In[20]:


model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape = shape))
model.add(MaxPool2D(2,2))
model.add(Conv2D(48,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dense(10,activation='softmax'))


# In[22]:


model.summary()


# In[26]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
x = model.fit(x_train,y_train,epochs =5,batch_size=128,validation_split=0.1)


# In[27]:


loss,accuracy = model.evaluate(x_test,y_test)
print(f'Accuracy:{accuracy*100}')


# In[50]:


fig = plt.figure(figsize=(10,3))
ax = fig.add_subplot(2,10,3,xticks=[],yticks=[])
ax.imshow(np.squeeze(x_test[3]),cmap='gray')


# In[51]:


predictions = model.predict(x_test)


# In[52]:


print(np.argmax(predictions[3]))


# In[ ]:




