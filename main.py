"""
SETUP
"""
### Import models
import pandas as pd
import numpy as np

### Import self-made functions
from CODE.data_preprocessing.split_val import split_val
from CODE.features.length_title import length_title
from CODE.features.field_variety import field_variety2
#from CODE.features.field_variety import field_variety
from CODE.features.team_size import team_size
from CODE.features.topic_variety import topics_variety
from CODE.features.venue_frequency import venue_frequency
from CODE.features.age import age

### Get the full train set:
data = pd.read_json('DATA/train-1.json')   # Numerical columns: 'year', 'references', 'citations'

### push the numerical columns to X and outcome to y
end = len(data)
num_X = data.loc[ 0:end+1 , ('doi', 'citations', 'year', 'references') ]


"""
FEATURE DATAFRAME: num_X

ALL: After writing a funtion to create a feature, please incorporate your new feature as a column on the dataframe below.
This is the dataframe we will use to train the models.
"""

### use feature function to create a new variable
title_len = length_title(data)      # returns: dictionary of lists: [doi](count)
field_var = field_variety2(data)    # returns: dictionary of lists: [doi](count)
team_sz = team_size(data)           # returns a numbered series
topic_var = topics_variety(data)    # returns a numbered series
venue_freq = venue_frequency(data)  # returns a dictionary: [venue](count)
paper_age = age(data)                     # returns a numbered series

### join the variables (type = series) to num_X 
num_X['team_size'] = team_sz
num_X['topic_variety'] = topic_var
num_X['age'] = paper_age

### join the variables (type = dictionary) to num_X
num_X['title_length'] = num_X['doi'].map(title_len)
num_X['field_variety'] = num_X['doi'].map(field_var)

### trainv/val split
X_train, X_val, y_train, y_val = split_val(num_X, target_variable = 'citations')



"""
INSERT split X and y on the train here
"""




"""
IMPLEMENT model fuctions here
"""