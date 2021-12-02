# Machine Learning Challenge
# Course: Machine Learning (880083-M-6)
# Group 58
 
##########################################
#             Import packages            #
##########################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##########################################
#      Import self-made functions        #
##########################################
from CODE.data_preprocessing.split_val import split_val
from CODE.data_preprocessing.find_outliers_tukey import find_outliers_tukey
from CODE.features.length_title import length_title
from CODE.features.field_variety import field_variety         
from CODE.features.field_popularity import field_popularity 
from CODE.features.team_size import team_size
from CODE.features.topic_variety import topics_variety
from CODE.features.topic_popularity import topic_popularity
from CODE.features.venue_popularity import venue_popularity
from CODE.features.venue_citations import venues_citations
from CODE.features.age import age
from CODE.features.author_database import author_database
from CODE.features.abst_words import abst_words
from CODE.features.author_h_index import author_h_index
from CODE.features.paper_h_index import paper_h_index


##########################################
#              Load datasets             #
##########################################
# Main datasets
data = pd.read_json('DATA/train.json')      # Training set
test = pd.read_json('DATA/test.json')       # Test set

# Author-centric datasets
#   These datasets were made using our self-made functions 'citations_per_author' (for the author_citation_dic)
#   These functions took a long time to make (ballpark ~10 minutes on a laptop in 'silent mode'), so instead we 
#   decided to run this function once, save the data, and reload the datasets instead of running the function again. 
import pickle
with open('my_dataset1.pickle', 'rb') as dataset:
    author_citation_dic = pickle.load(dataset)
with open('my_dataset2.pickle', 'rb') as dataset2:
    author_db = pickle.load(dataset2)


##########################################
#        Missing values handling         #
##########################################

# Missing values for feature 'fields_of_study'
data.loc[data['fields_of_study'].isnull(), 'fields_of_study'] = ""

# Missing values for feature 'title'
data.loc[data['title'].isnull(), 'title'] = ""

# Missing values for feature 'abstract'
data.loc[data['abstract'].isnull(), 'abstract'] = ""
    
# Missing values for features 'authors'
data.loc[data['authors'].isnull(), 'authors'] = ""

# Missing values for feature 'venue'
data.loc[data['venue'].isnull(), 'venue'] = ""
    
# Missing values for feature 'year'
# data.loc[data['fields_of_study'].isnull(), 'fields_of_study'] = mean(year) 
        #   Take mean by venue instead
        #       If venue not known, take something else?

# Missing values for feature 'references'
data.loc[data['references'].isnull(), 'references'] = ""

# Missing values for feature 'topics'
data.loc[data['topics'].isnull(), 'topics'] = ""

# Missing values for feature 'is_open_access'
data.loc[data['is_open_access'].isnull(), 'is_open_access'] = "" 
        #   Take most frequent occurrence for venue
        #       If venue not known, do something else?
    
##########################################
#       New variable, to add onto        #
##########################################
end = len(data)
num_X = data.loc[ 0:end+1 , ('doi', 'citations', 'year', 'references') ]  ##REMOVE DOI


"""
FEATURE DATAFRAME: num_X

ALL: After writing a funtion to create a feature, please incorporate your new feature as a column on the dataframe below.
This is the dataframe we will use to train the models.
"""

##########################################
#            Feature creation            #
##########################################
#### Add series of data to dataframe
"""
DO NOT change the order in this section if at all possible
"""
num_X['title_length'] = length_title(data)      # returns a numbered series
num_X['field_variety'] = field_variety(data)    # returns a numbered series 
num_X['field_popularity'] = field_popularity(data) # returns a numbered series
num_X['team_sz'] = team_size(data)           # returns a numbered series
num_X['topic_var'] = topics_variety(data)    # returns a numbered series
num_X['topic_popularity'] = topic_popularity(data) # returns a numbered series
num_X['venue_popularity'], num_X['venue'] = venue_popularity(data)  # returns a numbered series and a pandas.Series of the 'venues' column reformatted 
num_X['open_access'] = pd.get_dummies(data["is_open_access"], drop_first = True)  # returns pd.df (True = 1)
num_X['age'] = age(data)               # returns a numbered series. Needs to be called upon AFTER the venues have been reformed (from venue_frequency)
num_X['venPresL'] = venues_citations(data)   # returns a numbered series. Needs to be called upon AFTER the venues have been reformed (from venue_frequency)
keywords = ["method", "review", "randomized", "random control"]
num_X['has_keyword'] = abst_words(data, keywords)   #returns a numbered series: 1 if any of the words is present in the abstract, else 0

# Author H-index
author_db, reformatted_authors = author_database(data)
data['authors'] = reformatted_authors
num_X['h_index'] = paper_h_index(data, author_citation_dic) # Returns a numbered series. Must come after author names have been reformatted.

"""
END do not reorder
"""

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

# import json
#with open("sample.json", "w") as outfile:
    #json.dump(dictionary, outfile)