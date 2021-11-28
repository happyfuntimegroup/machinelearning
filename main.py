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

### Get the full train set:
data = pd.read_json('DATA/train-1.json')   # Numerical columns: 'year', 'references', 'citations'
test = pd.read_json('DATA/test.json')


"""
DEAL with missing values in "data" and "test" here - SELIN

doi --> ""
title --> ""
abstract --> "" 
authors --> [""]
venue --> ""
year --> mean of venue based on "data" ELSE from "data"
references --> 0  --think about this!
topic --> [""]
is_open-access --> base on venue ELSE from "data"
fields_of_study --> [""]
citations --> assume not blank

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




"""
IMPLEMENT regression models fuctions here
- exponential
"""

