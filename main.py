# Machine Learning Challenge
# Course: Machine Learning (880083-M-6)
# Group 58
 
##########################################
#             Import packages            #
##########################################
import numpy as np
import pandas as pd
import json
#import yake  #NOTE: with Anaconda: conda install -c conda-forge yake

##########################################
#      Import self-made functions        #
##########################################
from CODE.data_preprocessing.split_val import split_val
from CODE.data_preprocessing.find_outliers_tukey import find_outliers_tukey
from CODE.data_preprocessing.missing_values import missing_values1
from CODE.data_preprocessing.missing_values import missing_values2

#feature based on the title of the paper
from CODE.features.length_title import length_title

# features based on 'field_of_study' column 
from CODE.features.field_variety import field_variety         
from CODE.features.field_popularity import field_popularity
from CODE.features.field_citations_avarage import field_citations_avarage 

# features based on the topics of the paper
from CODE.features.topic_citations_avarage import topic_citations_avarage
from CODE.features.topic_variety import topics_variety
from CODE.features.topic_popularity import topic_popularity
from CODE.features.topic_citations_avarage import topic_citations_avarage

# features based on the abstract of the paper
from CODE.features.keywords import best_keywords
from CODE.features.abst_words import abst_words
from CODE.features.abst_words import abst_count

# features based on the venue of the paper
from CODE.features.venue_popularity import venue_popularity
from CODE.features.venue_citations import venues_citations

from CODE.features.age import age

# features based on the authors of the paper
from CODE.features.author_h_index import author_h_index
from CODE.features.paper_h_index import paper_h_index
from CODE.features.team_size import team_size
from CODE.features.author_database import author_database
print("Imports complete")

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
print("Data loaded")

##########################################
#       drop na rows          #
##########################################
data.dropna(axis='index', how = 'any')  # not helpful over missing_values.

##########################################
#        Missing values handling         #
##########################################
missing_values1(data)  # keep this on even with dropna; should be redundant but it seems to be needed...
missing_values1(test)  # this needs to stay on

##########################################
#       Create basic numeric df          #
##########################################
num_X = data.copy(deep = True)

##########################################
#            Feature creation            #
##########################################
"""
FEATURE DATAFRAME: num_X

ALL: After writing a funtion to create a feature, please incorporate your new feature as a column on the dataframe below.
This is the dataframe we will use to train the models.

DO NOT change the order in this section if at all possible
"""
num_X['title_length'] = length_title(data)      # returns a numbered series with wordlength of the title
test['title_length'] = length_title(test)       # returns a numbered series with wordlength of the title
num_X['field_variety'] = field_variety(data)    # returns a numbered series with amount of fields
test['field_variety'] = field_variety(test)    # returns a numbered series with amount of fields
num_X['field_popularity'], test['field_popularity'] = field_popularity(data, test) # returns a numbered series with 
num_X['field_citations_avarage'], test['field_citations_avarage']  = field_citations_avarage(data, test) # returns a numbered series
num_X['team_sz'] = team_size(data)           # returns a numbered series
test['team_sz'] = team_size(test)           # returns a numbered series
num_X['topic_variety'] = topics_variety(data)    # returns a numbered series
test['topic_variety'] = topics_variety(test)    # returns a numbered series
num_X['topic_popularity'], test['topic_popularity']= topic_popularity(data, test) # returns a numbered series
num_X['topic_citations_avarage'], test['topic_citations_avarage'] = topic_citations_avarage(data, test) # returns a numbered series
num_X['venue_popularity'], num_X['venue'], test['venue_popularity'], test['venue'] = venue_popularity(data, test)  # returns a numbered series and a pandas.Series of the 'venues' column reformatted 
num_X['open_access'] = pd.get_dummies(data["is_open_access"], drop_first = True)  # returns pd.df (True = 1)
test['open_access'] = pd.get_dummies(test["is_open_access"], drop_first = True)  # returns pd.df (True = 1)
num_X['age'] = age(data)               # returns a numbered series. Needs to be called upon AFTER the venues have been reformed (from venue_frequency)
test['age'] = age(test)               # returns a numbered series. Needs to be called upon AFTER the venues have been reformed (from venue_frequency)
num_X['venPresL'], test['venPresL'] = venues_citations(data, test)   # returns a numbered series. Needs to be called upon AFTER the venues have been reformed (from venue_frequency)
keywords = best_keywords(data, 1, 0.954, 0.955)    # from [data set] get [integer] keywords from papers btw [lower bound] and [upper bound] quantiles; returns list
num_X['has_keyword'] = abst_words(data, keywords)#returns a numbered series: 1 if any of the words is present in the abstract, else 0
test['has_keyword'] = abst_words(test, keywords)#returns a numbered series: 1 if any of the words is present in the abstract, else 0
num_X['keyword_count'] = abst_count(data, keywords) # same as above, only a count (noot bool)
test['keyword_count'] = abst_count(test, keywords) # same as above, only a count (noot bool)

# Author H-index
author_db, data['authors'] = author_database(data)
_, test['authors'] = author_database(test) # reformatting authors name from test database
num_X['h_index'], test['h_index'] = paper_h_index(data, author_citation_dic, test) # Returns a numbered series. Must come after author names have been reformatted.


"""
END do not reorder
"""
print("Features created")

##########################################
#    Filling specific missing values     #
##########################################
missing_values2(num_X)
missing_values2(test)

### Drop columns containing just strings or boolean
num_X = num_X.drop( ['authors', 'abstract', 'topics', 'title', 'venue', 'fields_of_study', 'is_open_access', 'doi'], axis = 1)
test = test.drop(   ['authors', 'abstract', 'topics', 'title', 'venue', 'fields_of_study', 'is_open_access'], axis = 1)

### Drop duplicate rows - this is a lenient definition, but it returns a very small number of rows, so I like it better than a full match 
duplicate = data[data.duplicated(['title', 'year'])]
a = list(duplicate.index)
num_X = num_X.drop(labels = a)

print("Missing values handled")
      
##########################################
#      log transform some columns        #
##########################################
# Any columns that we build log transformations for should have no zeros and give real values to nans
# Also we would have to convert y back from log for the predict: there is somehing about this in one of the documents somewhere...

## Fill zeros
a = num_X['team_sz'].median()
num_X.loc[num_X['team_sz'] == 0, 'team_sz'] = a  # fills 13 zero values
test.loc[test['team_sz'] == 0, 'team_sz'] = a  

b = num_X['references'].median()
num_X.loc[num_X['references'] == 0, 'references'] = a  # fills 13 zero values
test.loc[test['references'] == 0, 'references'] = a 

## Log some columns
num_X['nlog_year'] = np.log(num_X['year'])
num_X['nlog_title_length'] = np.log(num_X['title_length'])
num_X['nlog_field_citations_avarage'] = np.log(num_X['field_citations_avarage'])
num_X['nlog_topic_popularity'] = np.log(num_X['topic_popularity'].astype(np.float64))
num_X['nlog_venue_popularity'] = np.log(num_X['venue_popularity'].astype(np.float64))
num_X['nlog_team_sz'] = np.log(num_X['team_sz'].astype(np.float64)) 
num_X['nlog_references'] = np.log(num_X['references'].astype(np.float64)) 

test['nlog_year'] = np.log(test['year'])
test['nlog_title_length'] = np.log(test['title_length'])
test['nlog_field_citations_avarage'] = np.log(test['field_citations_avarage'])
test['nlog_topic_popularity'] = np.log(test['topic_popularity'].astype(np.float64))
test['nlog_venue_popularity'] = np.log(test['venue_popularity'].astype(np.float64))
test['nlog_team_sz'] = np.log(test['team_sz'].astype(np.float64)) 
test['nlog_references'] = np.log(test['references'].astype(np.float64)) 

"""
END do not reorder
"""
print("Features created")


##########################################
#   fix rows with 0 in logged columns    #
##########################################
print(num_X.shape)
num_X = num_X.replace([np.inf, -np.inf], np.nan)
num_X = num_X.dropna(axis='index', how = 'any')  
print(num_X.shape)

##########################################
#            lable encode y: log         #
##########################################
# Need to add a constant to avoid zeros - REMEMBER this when undoing the log.
num_X['citations'] = (np.log(num_X['citations'].astype(np.float64)+2))

##########################################
#            Train/val split             #
##########################################
X_train, X_val, y_train, y_val = split_val(num_X, target_variable = 'citations')
print("Data split")

##########################################
#   Outlier detection X_trian: Quantile  #
##########################################
### MODEL code for outlier detection
# You still want as many validation samples as possible to get a realistic idea of model performance.
#   Therefore, removing outliers happens after splitting the dataset into a training and validation set.
#   After splitting, we detect outliers in the training set to give the model a general idea of what to expect.

out_ref = (find_outliers_tukey(x = X_train['references'], top = 95, bottom = 0))[0]
out_team = (find_outliers_tukey(x = X_train['team_sz'], top = 95, bottom = 0))[0]
out_tvar = (find_outliers_tukey(x = X_train['topic_variety'], top = 95, bottom = 0))[0]
out_ven = (find_outliers_tukey(x = X_train['venPresL'], top = 95, bottom = 0))[0]
out_h = (find_outliers_tukey(x = X_train['h_index'], top = 95, bottom = 0))[0]
out_cit = (find_outliers_tukey(x = y_train['citations'], top = 93, bottom = 0))[0]

out_rows = out_cit + out_ref + out_team + out_tvar + out_ven + out_h

# Potential features to get rid of: team_sz; year and age are perfect correlates
print("Outliers handeled")

##########################################
#         Model implementations          #
##########################################
# It takes 30 to 60 minutes to run all models

from CODE.models.regression import simple_linear
from CODE.models.regression import log_reg
from CODE.models.regression import sdg_reg
from CODE.models.regression import poly_reg
from CODE.models.regression import pois_reg
from CODE.models.non_linear import de_tree_reg
from CODE.models.non_linear import kn_reg
from CODE.models.non_linear import my_svr
from CODE.models.non_linear import mlp_reg
"""
IMPLEMENT models here: to run a model, delete the # and run
"""

#----------- Check for changes
check_X = X_train.copy(deep = True)
check_y = y_train.copy(deep = True)

#-----------simple regression, all columns
# Leave this on as a baseline
# model = simple_linear(X_train.drop(labels = out_rows), y_train.drop(labels = out_rows), X_val, y_val)

"""
MODEL RESULTS:
r2: 0.04709627575317288
MSE: 33.38996
# Worse after extra outlier removal (0.015478)
"""
#-----------logistic regression, all columns
#model = log_reg(X_train.drop(labels = out_rows), y_train.drop(labels = out_rows), X_val, y_val)

"""
MODEL RESULTS:
R2: 0.006551953988217396
MSE: 34.07342328208346
# Worse after extra outlier removal (0.003)
"""

#-----------SGD regression, all columns
# model = sdg_reg (X_train.drop(labels = out_rows), y_train.drop(labels = out_rows), X_val, y_val)

"""
lr = [ 1, .1, .01, .001, .0001]
learning_rate in ['constant', 'optimal', 'invscaling']:
loss in ['squared_error', 'huber']:

# MODEL RESULTS:
# Best outcome, before extra outlier removal: ('constant', 0.01, 'squared_error', 35.74249957361433, 0.04476790061780822)
# Best outcome after extra outlier removal: ('constant', 0.01, 'squared_error', 37.08290449479669, 0.019303736163186702)
"""

#-----------polynomial regression, all columns
#model = poly_reg (X_train.drop(labels = out_rows), y_train.drop(labels = out_rows), X_val, y_val, 3)

"""
MODEL RESULTS:
r2: -0.05109 (degree = 2)
r2: -0.0378 (degree = 3)
r2: -5.5816 (degree = 4)
MAE 35.1660
"""

#-----------poisson regression, all columns
# pois_reg (X_train.drop(labels = out_rows), y_train.drop(labels = out_rows), X_val, y_val)

"""
MODEL RESULTS:
r2: 0.022145
MAE: 39.21127
"""

#-----------simple linear regression, dropping columns

"""
USE this code to run one of the simple regression models, successively dropping one column
To run, unhash the full function, then unhash the specific model
For a baseline, run the corresponding model above
"""
# summaries = list(X_train.columns)
# print(summaries)

# for i in range(len(summaries)):
#     X_train_small = X_train.copy()
#     X_val_small = X_val.copy()
#     drops = summaries[i]
#     X_train_small.drop(drops, inplace = True, axis=1)
#     X_val_small.drop(drops, inplace = True, axis=1)

#     print("dropped:", summaries[i])
    
#     #simple_linear(X_train_small, y_train, X_val_small, y_val)  #dropping venue_popularity helps a tiny bit
#     #log_reg(X_train_small, y_train, X_val_small, y_val)


#----------- Random Forrest for Regression
# +/- 5 min to run
#model = de_tree_reg (X_train.drop(labels = out_rows), y_train.drop(labels = out_rows), X_val, y_val, 50)

"""
MODEL RESULTS:
r2: 0.03378200504507167
"""

#----------- K-Neighbors for Regression
#model = kn_reg (X_train.drop(labels = out_rows), y_train.drop(labels = out_rows), X_val, y_val)

"""
MODEL RESULTS:
r2: 0.05784278300196066
"""

#----------- SVR
# from sklearn.svm import SVR
# svr = SVR()
# model = svr.fit(X_train.drop(labels = out_rows), np.ravel(y_train.drop(labels = out_rows) ))
# r_sq1 = model.score(X_val, y_val)
# print('r2 scr:', r_sq1)


#-----------  Multi-layer Perceptron for Regression
#model = mlp_reg (X_train.drop(labels = out_rows), y_train.drop(labels = out_rows), X_val, y_val) 

"""
OPTIONS:
activation= 'identity', 'logistic', 'tanh', 'relu'
solver= 'lbfgs', 'sgd', 'adam'
lr= 'constant', 'invscaling', 'adaptive'
DEFAULT values: maxit=500, activation='relu', solver='adam', alpha=0.0001, lr='constant'

MODEL RESULTS:
r2: 0.48312678097660333
"""

print("data unchanged:")
print(check_X.equals(X_train))
print(check_y.equals(y_train))
print("Models complete")

##########################################
#     Fit model to X_train and X_val     #
##########################################
X_total = pd.concat([X_train, X_val])
y_total = pd.concat([y_train, y_val])

out_ref = (find_outliers_tukey(x = X_total['references'], top = 95, bottom = 0))[0]
out_team = (find_outliers_tukey(x = X_total['team_sz'], top = 95, bottom = 0))[0]
out_tvar = (find_outliers_tukey(x = X_total['topic_variety'], top = 95, bottom = 0))[0]
out_ven = (find_outliers_tukey(x = X_total['venPresL'], top = 95, bottom = 0))[0]
out_h = (find_outliers_tukey(x = X_total['h_index'], top = 95, bottom = 0))[0]
out_cit = (find_outliers_tukey(x = y_total['citations'], top = 93, bottom = 0))[0]

out_rows = out_cit + out_ref + out_team + out_tvar + out_ven + out_h

model.fit(X_total.drop(labels = out_rows), np.ravel(y_total.drop(labels = out_rows)))

##########################################
#  Writing file with predicted values    #
##########################################
"""
    Creates new DataFrame with DOI of the papers, 
    and predicted citation values.
"""

df_output = pd.DataFrame(columns = ['doi','citations'])

y_test_log = model.predict(test.drop(['doi'], axis=1))
y_test = np.exp(y_test_log) - 2

# If a model is used where the y (target) did NOT need to be raveled, use this code
for index, i_paper in test.iterrows():
    df_output.loc[index, 'doi'] = i_paper['doi'] 
    df_output.loc[index, 'citations'] = y_test[index]
   
# If a model is used where the y (target) had to be raveled (such as MLPRegressor or SVR), use this code
for index, i_paper in test.iterrows():
    df_output.loc[index, 'doi'] = i_paper['doi'] 
    df_output.loc[index, 'citations'] = y_test[index][0]

list_dic_output = df_output.to_dict(orient = 'records')

#jsonOutput = json.dumps(list_dic_output, indent = 4)
with open('OUTPUT/predicted.json', 'w') as f:
    json.dump(list_dic_output, f)
