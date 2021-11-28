"""
SETUP
"""
### Import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

### Import self-made functions
from CODE.data_preprocessing.split_val import split_val
from CODE.data_preprocessing.find_outliers_tukey import find_outliers_tukey
from CODE.features.length_title import length_title
from CODE.features.field_variety import field_variety2
#from CODE.features.field_variety import field_variety
from CODE.features.team_size import team_size
from CODE.features.topic_variety import topics_variety
from CODE.features.venue_frequency import venue_frequency
from CODE.features.venue_citations import venues_citations
from CODE.features.age import age
#from CODE.features.author_database import author_database
#from CODE.features.author_name import author_name
from CODE.features.abst_words import abst_words

### Load all datasets:
data = pd.read_json('DATA/train-1.json')   # Numerical columns: 'year', 'references', 'citations'
test = pd.read_json('DATA/test.json')

import pickle
with open('my_dataset1.pickle', 'rb') as data:
    author_citation_dic = pickle.load(data)
with open('my_dataset2.pickle', 'rb') as data:
    author_db = pickle.load(data)


"""
###DEAL with missing values in "data" and "test" here 

data = data   #data = test for test data
dict_field_var = {}
dict_title = {}
dict_abstract = {}
dict_authors = {}
dict_venue = {}
dict_year = {}
dict_references = {}
dict_topics = {}
dict_access = {}
dict_citations = {} #delete this for test data

for i in range(len(data)):
    doi = data.iloc[i]['doi'] 
    fields = data.iloc[i]['fields_of_study']
    title = data.iloc[i]['title']
    abstract = data.iloc[i]['abstract']
    authors = data.iloc[i]['authors']
    venue = data.iloc[i]['venue']
    year = data.iloc[i]['year']
    references = data.iloc[i]['references']
    topics = data.iloc[i]['topics']
    access = data.iloc[i]['is_open_access']
    citations = data.iloc[i]['citations'] #delete for test data
        
#   if doi == None:
#       doi = i
    if fields == None:    #Filling the categorical value with a new type for the missing values.
        fields = "None"    
    dict_field_var[doi] = (len(fields))
    if title == None: 
       title = "None"
    dict_title[doi] = (len(title))
    if abstract == None:
        abstract = "None" # abstract = title
    dict_abstract[doi] = (len(fields))
    if authors == None:
        authors = "None"
    dict_authors[doi] = (len(authors))
    if venue == None:          #Filling the missing data with mode if it’s a categorical value?
        venue = "None"
    dict_venue[doi] = (len(venue))
    if year == None:
        year = mean(year)   #mean of year based on "data" 
    dict_year[doi] = (len(year))
    if references == None:
        references = mean(references) # references = 999
    dict_references[doi] = (len(references))
    if topics == None:
        topics = "None" #topic = title
    dict_topics[doi] = (len(references))
    if access == None:
        access = "None"   #based on venue?
    dict_access[doi] = (len(access))
    if citations == None:           #delete for test data
        citations = "None"
    dic_citations[doi] =(len(citations))    
    
return dict_field_var 
return dict_title 
return dict_abstract 
return dict_authors 
return dict_venue
return dict_year 
return dict_references
return dict_topic 
return dict_access
return dict_citations #delete for test data
"""



### push the numerical columns to num_X
end = len(data)
num_X = data.loc[ 0:end+1 , ('doi', 'citations', 'year', 'references') ]  ##REMOVE DOI




"""
FEATURE DATAFRAME: num_X

ALL: After writing a funtion to create a feature, please incorporate your new feature as a column on the dataframe below.
This is the dataframe we will use to train the models.
"""

### use feature function to create a new variable
"""
DO NOT change the order in this section if at all possible
"""
title_len = length_title(data)      # returns: dictionary of lists: [doi](count)
field_var = field_variety2(data)    # returns: dictionary of lists: [doi](count)
team_sz = team_size(data)           # returns a numbered series
topic_var = topics_variety(data)    # returns a numbered series
venue_db, venues_reformatted = venue_frequency(data)  # returns a dictionary: [venue](count) and a pandas.Series of the 'venues' column reformatted 
num_X['venue'] = venues_reformatted # Dataframe needs a venue to deal with missing values
open_access = pd.get_dummies(data["is_open_access"], drop_first = True)  # returns pd.df (True = 1)
paper_age = age(data)               # returns a numbered series. Needs to be called upon AFTER the venues have been reformed (from venue_frequency)
venPresL = venues_citations(data)   # returns a numbered series. Needs to be called upon AFTER the venues have been reformed (from venue_frequency)
keywords = ["method", "review", "randomized", "random control"]
abst_keywords = abst_words(data, keywords)   #returns a numbered series: 1 if any of the words is present in the abstract, else 0
"""
END do not reorder
"""


### join the variables (type = series) to num_X 
num_X['team_size'] = team_sz
num_X['topic_variety'] = topic_var
num_X['age'] = paper_age
num_X['open_access'] = open_access
num_X['has_keyword'] = abst_keywords
num_X['venue'] = venues_reformatted
num_X['venPresL'] = venPresL

### join the variables (type = dictionary) to num_X
num_X['title_length'] = num_X['doi'].map(title_len)
num_X['field_variety'] = num_X['doi'].map(field_var)


# Check venue and add venue_frequency to each paper
venue_freq = pd.Series(dtype=pd.Int64Dtype())
for index, i_paper in num_X.iterrows():
    venue_freq[index,] = venue_db[i_paper['venue']] 
num_X['venue_freq'] = venue_freq


### Drop columns containing just strings
num_X = num_X.drop(['venue', 'doi'], axis = 1)


## train/val split
X_train, X_val, y_train, y_val = split_val(num_X, target_variable = 'citations')


"""
INSERT outlier detection on X_train here - ALBERT
"""
### MODEL code for outlier detection
### names: X_train, X_val, y_train, y_val

# print(list(X_train.columns))

# out_y = (find_outliers_tukey(x = y_train['citations'], top = 93, bottom = 0))[1]
# out_X = (find_outliers_tukey(x = X_train['references'], top = 85, bottom = 0))[1]
# out_rows = out_y + out_X
# out_rows = sorted(list(set(out_rows)))

# print("y:", out_y)
# print("X:", out_X)
# print("rows:", out_rows)


"""
IMPLEMENT regression models fuctions here
- exponential
"""

