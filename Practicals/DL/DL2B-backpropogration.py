#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.DataFrame([[8,8,4],[7,9,5],[6,10,6],[5,12,7]],columns=['cgpa','profile_score','lpa'])


# In[3]:


df


# In[4]:


import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense


# In[5]:


model = Sequential()

model.add(Dense(2,activation='linear',input_dim=2))
model.add(Dense(1,activation='linear'))


# In[6]:


model.summary()


# In[7]:


model.get_weights()


# In[8]:


new_weights = [np.array([[0.1,0.1],[0.1,0.1]],dtype=np.float32),
               np.array([0.,0.],dtype=np.float32),
               np.array([[0.1],[0.1]],dtype=np.float32),
               np.array([0.],dtype=np.float32)]


# In[9]:


model.set_weights(new_weights)


# In[10]:


model.get_weights()


# In[11]:


optimizer =  keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error',optimizer=optimizer)


# In[12]:


model.fit(df.iloc[:,0:-1].values,df['lpa'].values,epochs=75,verbose=1,batch_size=1)


# In[18]:


model.get_weights()


# In[ ]:




