#!/usr/bin/env python
# coding: utf-8

# In[ ]:

"""
Input: dataframe, list of keywords
Output: panda series
abst_words returns a series of booleans: was any one of the keywords in that paper's abstract
abst_count returns a series of integers: how many of the keywords were in that paper's abstract
"""
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
                    flag += 1
            abst_key.append(flag)
    return pd.Series(abst_key)