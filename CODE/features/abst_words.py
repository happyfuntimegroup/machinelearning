#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def abst_words (the_data, keywords):
    import pandas as pd
    
    abst = the_data['abstract']
    abst_key = []    
    
    for i in abst:
        if i == None:
            abst_key.append(0)
            continue
        else:
            flag = 0
            for word in keywords:
                if word in i.lower():
                    flag = 1
            abst_key.append(flag)
    return pd.Series(abst_key)

def abst_count (the_data, keywords):
    abst = the_data['abstract']
    abst_key = []    
    
    for i in abst:
        if i == None:
            abst_key.append(0)
            continue
        else:
            flag = 0
            for word in keywords:
                if word in i.lower():
                    flag += 1
            abst_key.append(flag)
    return pd.Series(abst_key)