#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import pandas as pd


# In[4]:


#import json

#with open(r"C:\Users\SelinZ\OneDrive\Desktop\ML\train-1.json") as f:
 #   traind = json.load(f)

#df = pd.DataFrame.from_dict(traind)


# In[12]:


#The number of different topics that a certain article is about 

def topics_variety(df):
    """
    Computes the number of different topics that each paper is about by taking the number of topics in 'topics'
    Input:
        - df['topics']:    dataframe (dataset); 'topics' column                           [pandas dataframe]
    Output:
        - Topic variety:   vector of topic_variety for each paper of the given dataset      [pandas series]
                            with Topic_variety                                                   [int]
    """
    import pandas as pd

    Topic_variety = pd.Series([len(i) for i in df['topics']])      # Topic variety

    return Topic_variety                                           #Output


# In[18]:


#topics_variety(df).value_counts()

