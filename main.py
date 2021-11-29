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
#from CODE.features.field_variety import field_variety2
#from CODE.features.field_variety import field_variety         # Not working anymore?
from CODE.features.team_size import team_size
from CODE.features.topic_variety import topics_variety
from CODE.features.venue_frequency import venue_frequency
from CODE.features.venue_citations import venues_citations
from CODE.features.age import age
#from CODE.features.author_database import author_database
#from CODE.features.author_name import author_name
from CODE.features.abst_words import abst_words

### Load all datasets:
data = pd.read_json('DATA/train.json')   # Numerical columns: 'year', 'references', 'citations'
test = pd.read_json('DATA/test.json')

import pickle
with open('my_dataset1.pickle', 'rb') as dataset:
    author_citation_dic = pickle.load(dataset)
with open('my_dataset2.pickle', 'rb') as dataset:
    author_db = pickle.load(dataset)


"""
###DEAL with missing values in "data" and "test" here 

data = data  

#dict_field_num = {}
dict_fields = {}
for i in range(len(data)):
    doi = data.iloc[i]['doi'] 
    fields = data.iloc[i]['fields_of_study'] 
    if fields == None:   
        fields = ""   #when we put "None" here, counts its characters and gives 4 for an empty value instead of 0
    #dict_field_num[doi] = len(fields) #double check field_variety2 function and the need for a function like that
    dict_fields[doi] = fields  
#dict_field_num.values() 
#dict_fields.values() 


dict_title = {}
#dict_length_title = {}
for i in range(len(data)):
    doi = data.iloc[i]['doi'] 
    title = data.iloc[i]['title']
    if title == None: 
        title = ""
    dict_title[doi] = (title)
   # dict_length_title[doi] = len(title)
    
#dict_title.values() 
#dict_length_title.values()


dict_abstract = {}
#dict_length_abstract = {}
for i in range(len(data)):
    doi = data.iloc[i]['doi'] 
    abstract = data.iloc[i]['abstract']
    if abstract == None:
        abstract = "" # abstract = title?
    dict_abstract[doi] = (abstract)
 #   dict_length_abstract[doi] = len(abstract)
    
#dict_abstract.values() 
#dict_length_abstract.values()

dict_authors = {}
#dict_author_num = {}
for i in range(len(data)):
    doi = data.iloc[i]['doi'] 
    authors = data.iloc[i]['authors']
    if authors == None:
        authors = ""
    dict_authors[doi] = (authors)
 #   dict_author_num[doi] = len(authors)
    
#dict_authors.values() 
#dict_author_num.values()


dict_venues = {}
for i in range(len(data)):
    doi = data.iloc[i]['doi'] 
    venue = data.iloc[i]['venue']
    if venue == None:
        venue = ""
    dict_venues[doi] = (venue)
    
#dict_venues.values() 


dict_year = {}
for i in range(len(data)):
    doi = data.iloc[i]['doi'] 
    year = data.iloc[i]['year'] #change it based on venue?
    if year == None:
        year = mean(year)
    dict_year[doi] = (year)
    
#dict_year.values() 


dict_references = {}
for i in range(len(data)):
    doi = data.iloc[i]['doi'] 
    references = data.iloc[i]['references']
    if references == None:      
        references = ""     #999?
    dict_references[doi] = (references)
    
#dict_references.values()


dict_topics = {}
#dict_topics_num ={}
for i in range(len(data)):
    doi = data.iloc[i]['doi'] 
    topics = data.iloc[i]['topics']
    if topics == None:
        topics = [""]         #topic = title?
    dict_topics[doi] = (topics)
 #   dict_topics_num[doi] = len(topics)
    
#dict_topics.values() 
#dict_topics_num.values()


dict_access = {}
for i in range(len(data)):
    doi = data.iloc[i]['doi'] 
    access = data.iloc[i]['is_open_access']
    if access == None:
        access = "" 
    dict_access[doi] = (access)
    
#dict_access.values() 


dict_citations = {}
for i in range(len(data)):
    doi = data.iloc[i]['doi'] 
    citations = data.iloc[i]['citations']
    if citations == None:
        citations = 9999 
    dict_citations[doi] = (citations)
    
#dict_citations.values()        
        
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
#field_var = field_variety2(data)    # returns: dictionary of lists: [doi](count)
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
#num_X['field_variety'] = num_X['doi'].map(field_var)


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

# out_y = (find_outliers_tukey(x = y_train['citations'], top = 93, bottom = 0))[0]
# out_X = (find_outliers_tukey(x = X_train['team_size'], top = 99, bottom = 0))[0]
# out_rows = out_y + out_X
# out_rows = sorted(list(set(out_rows)))

# print("X_train:")
# print(X_train.shape)
# X_train = X_train.drop(labels = out_rows)
# print(X_train.shape)
# print()
# print("y_train:")
# print(y_train.shape)
# y_train = y_train.drop(labels = out_rows)
# print(y_train.shape)

"""
IMPLEMENT regression models fuctions here
- exponential
"""

