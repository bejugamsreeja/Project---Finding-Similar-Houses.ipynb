#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
house_data = pd.read_csv('https://raw.githubusercontent.com/zekelabs/data-science-complete-tutorial/master/Data/house_rental_data.csv.txt',index_col='Unnamed: 0')
house_data.head()


# In[8]:


#Without Scaling
#Sqft & Price gets undue advantage


from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=3)
nn.fit(house_data)


# In[9]:


nn.kneighbors(house_data[:1])


# In[10]:


house_data.iloc[[0,354,319]]


# In[11]:


#Scaling to bring every feature in same scale
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()
house_data_scaled = minmax.fit_transform(house_data.values)
nn.fit(house_data_scaled)


# In[12]:


nn.kneighbors(house_data_scaled[:1])


# In[13]:


house_data.iloc[[0,178,330]]


# In[ ]:




