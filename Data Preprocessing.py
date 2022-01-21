#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[3]:


IDs = []
for file in os.listdir('/Users/manny/Downloads/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/set_a'):
    if file.endswith('.txt'):
        IDs.append(file)
        


# In[4]:


time_dfs = []
static_dfs = []

# use IDs list to make dataframes for each txt file
for i in IDs:
    # use read_csv to make dataframe
    df = pd.read_csv('{}'.format(i))
    
    # make one dataframe with just temporal values
    df['RecordID'] = df.at[0, 'Value']
    df1 = df[6:]
    
    # make another dataframe with just static values
    d = df[:6]
    values = d['Value'].values    
    df2 = pd.DataFrame([values], columns=['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight'])
    
    # append lists with dataframes
    time_dfs.append(df1)
    static_dfs.append(df2)
    
# Concatenate all created dataframes into one
df_time = pd.concat(time_dfs).reset_index(drop=True)
df_static = pd.concat(static_dfs).reset_index(drop=True)


# In[24]:


df_static.shape


# In[25]:


df_time.shape


# In[7]:


# create function to replace missing/erroneous values in columns

def data_preprocessing(series, ok_range):
    col = series.copy()
    good_values = []
    indexes = []
    
    # append lists with good values and bad indexes
    for i in series:
        # if the metric is between an accepted range, use to calc mean
        # ex: 7'5" is max height, so if i is < 7'5" it is added to list
        if i >= ok_range[0] and i <= ok_range[1]:
            good_values.append(i)
            
        # if out of range, replace with -5    
        else:
            x = col[col == i].index
            indexes.append(x)
    
    # replace series values with -5
    for x in indexes:
        series[x] = -5
        
    # now replace bad values with mean for column
    mean = sum(good_values) / len(good_values)
    mean = round(mean, 1)
    new_series = series.replace(-5, mean)
    return new_series
            
            


# In[28]:


# create function to replace missing/erroneous values in columns

def data_preprocessing2(series, ok_range):
    col = series.copy()
    good_values = []
    indexes = []
    
    # append lists with good values and bad indexes
    for i in series:
        # if the metric is between an accepted range, use to calc mean
        # ex: 7'5" is max height, so if i is < 7'5" it is added to list
        if i >= ok_range[0] and i <= ok_range[1]:
            good_values.append(i)
            
        # if out of range, replace with -5    
        else:
            x = col[col == i].index
            indexes.append(x)
    
    # replace series values with -5
    for x in indexes:
        series[x] = -5
        
    # now replace bad values with mean for column
    new_series = series.replace(-5, np.nan)
    return new_series
            


# In[8]:


height_col = data_preprocessing(df_static['Height'], [140, 230])


# In[9]:


weight_col = data_preprocessing(df_static['Weight'], [40, 301])


# In[29]:


gender_col = data_preprocessing2(df_static['Gender'], [0, 1])


# In[30]:


ICU_col = data_preprocessing2(df_static['ICUType'], [1, 4])


# In[10]:


df_static['Height'] = height_col


# In[11]:


df_static['Weight'] = weight_col


# In[27]:


df_static['Gender'] = gender_col


# In[32]:


df_static['ICUType'] = ICU_col


# In[37]:


df_static = df_static.iloc[:, :6]


# In[52]:


df_static.shape


# In[39]:


df_static.to_csv('pred_mor_static.csv')


# In[ ]:





# In[15]:


def time_to_hours(series):
    series = series.copy()
    lols = []
    
    for i in series:
        j = int(i[:2])
        k = int(i[-2:])
        l = j + (k / 60)
        lols.append(round(l, 2))    
    return lols
            
            


# In[16]:


hours = pd.Series(time_to_hours(df_time['Time']))


# In[18]:


df_time['Time'] = hours


# In[40]:


df_time.head()


# In[53]:


df_time.Parameter.value_counts


# In[20]:


df_time.to_csv('pred_mor_time.csv')


# In[ ]:





# In[41]:


df1 = df_time[df_time['RecordID'] == 132592]


# In[50]:


df1l.max()


# In[48]:


print(df1[:50])


# # Preparing Test Set

# In[ ]:


test_IDs = []
for file in os.listdir('/Users/manny/Downloads/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/Outcomes-a.txt'):
    if file.endswith('.txt'):
        IDs.append(file)


# In[54]:


df_test = pd.read_csv('/Users/manny/Downloads/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/Outcomes-a.txt')


# In[55]:


df_test


# In[57]:


outcomes = df_test.drop(['SAPS-I', 'SOFA', 'Length_of_stay', 'Survival'], axis=1)


# In[58]:


outcomes


# In[59]:


outcomes = outcomes.sort_values(by=['RecordID'])


# In[60]:


outcomes.to_csv('outcomes_a.csv')


# In[ ]:




