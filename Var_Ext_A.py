#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from functools import reduce


# In[2]:


df_static = pd.read_csv('pred_mor_static.csv')

df_static.drop('Unnamed: 0', axis=1, inplace=True)
df_static.head()


# In[3]:


print(len(df_static['RecordID'].unique()))


# In[4]:


df_time = pd.read_csv('pred_mor_time.csv')
df_time.drop('Unnamed: 0', axis=1, inplace=True)


# In[5]:


df_time.index[df_time['Parameter'] == 'ICUType'].tolist()


# In[6]:


df_time = df_time[df_time.Parameter != 'Height']
df_time = df_time[df_time.Parameter != 'Gender']
df_time = df_time[df_time.Parameter != 'ICUType']


# In[7]:


df_static[~df_static.RecordID.isin(df_time.RecordID)]


# In[8]:


dftime_order = df_time.sort_values(by=['RecordID', 'Time'])


# In[9]:


print(len(dftime_order['RecordID'].unique()))


# In[10]:


dfstatic_order = df_static.sort_values(by=['RecordID'])


# In[11]:


row_index = sorted(df_time['RecordID'].unique())
print(len(row_index))


# In[12]:


params = sorted(df_time['Parameter'].unique())
params


# In[13]:


def extract_var(df, params, indexes):
    df = df.copy()
    
    # create empty lists
    col_headers = []
    dflst = []
    datalst = []
    dfss = []
    
    for x in params:
        col_head = []
        # create new df containing only this specific param
        new_df = df[df.iloc[:, 1] == x]
        # append list with new df
        dflst.append(new_df)
        
        # append col_head list with headers for stats for this param
        col_head.append('RecordID')
        col_head.append('{}_first'.format(x))
        col_head.append('{}_last'.format(x))
        col_head.append('{}_min'.format(x))
        col_head.append('{}_max'.format(x))
        col_head.append('{}_q1'.format(x))
        col_head.append('{}_median'.format(x))
        col_head.append('{}_q3'.format(x))
        col_head.append('{}_mean'.format(x))
        col_head.append('{}_count'.format(x))
        
        col_headers.append(col_head)
    
    # loop through dflst to get stats for each df
    for y in dflst:
        # make list where each ID occurs only once
        uni_ids = y['RecordID'].unique()
        
        # create list to be filled with stats lists
        lst = []
        # for each recordID/patient, get the first, last, max, min
            # mean, median, q1, q2, count values where possible
        for z in uni_ids:
            # create list to be filled with stats
            idl = []
            
            # calc stats values
            patient = y[y['RecordID'] == z]   
            first = patient['Time'].idxmin
            last = patient['Time'].idxmax
            minv = patient['Value'].min()
            maxv = patient['Value'].max()
            q1 = patient['Value'].quantile(0.25)
            median = patient['Value'].quantile(0.5)
            q3 = patient['Value'].quantile(0.75)
            mean = patient['Value'].mean()
            count = patient['Value'].count()
            iqr = q3 - q1

            # append list with descriptive stats
            idl.append(z)
            idl.append(patient['Value'][first])
            idl.append(patient['Value'][last])
            if minv < (q1 - (1.5*iqr)):
                idl.append(q1 - 0.5*iqr)
            else:
                idl.append(minv)
            if maxv > (q3 + (1.5*iqr)):
                idl.append(q3 + 0.5*iqr)
            else:
                idl.append(maxv)
            idl.append(q1)
            idl.append(median)
            idl.append(q3)
            idl.append(mean)
            idl.append(count)
            # append lst with list of stats
            lst.append(idl)
        # append datalst with list of lists of stats for each unique ID    
        datalst.append(lst)
    
    # use enumerate to make use of indexes and create new dfs of stats
        # with headers
    for u,v in enumerate(col_headers):
        stats = np.array(datalst[u])
        new_df = pd.DataFrame(data=stats, columns=v)
        # append dfss list with dfs for each parameter
        dfss.append(new_df)
        

    # concatenate with outer join to keep all record IDs for each var
        # will fill patients with no values for certain records w NaN
    ext_var = reduce(lambda x, y: pd.merge(x, y, how='outer', on='RecordID'), dfss)
 
    # return dataframe into new variable
    return ext_var


# In[14]:


df_extvars = extract_var(dftime_order, params, row_index)


# In[15]:


print("A: {}".format(df_extvars.shape))


# In[16]:


dfstatic = dfstatic_order.astype('float')


# In[17]:


df_extvars.dtypes


# In[18]:


df_extvar = df_extvars.sort_values(by=['RecordID'])


# In[19]:


final_stat = pd.merge(dfstatic, df_extvar, how='outer', on='RecordID')


# In[20]:


final_stat.to_csv('pred_ready.csv')


# In[ ]:




