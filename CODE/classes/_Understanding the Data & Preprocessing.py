import pandas as pd
import seaborn as sns #for EDA
import numpy as np


#Before having split it into train & validation

import json

with open(r"C:\Users\SelinZ\OneDrive\Desktop\ML\train-1.json") as f:
    traind = json.load(f)

df = pd.DataFrame.from_dict(traind)
df

pd.DataFrame(df.items())


df.dtypes

#mean,std and max for references & citations
df.describe()

#missing values- How shall we deal with them? 

df.isnull().sum().sort_values(ascending= False)
#fields_of_study    136
#year                 3
#abstract           159

sns.relplot(x= 'citations', y = 'references', data = df)
sns.relplot(x= 'citations', y = 'references', data = df, hue = 'is_open_access')

sns.relplot(x= 'citations', y = 'year', data = df)

p_df = df.drop_duplicates('venue')
p_df = p_df.sort_values('citations', ascending=False)
p_df = p_df.head(50)
p_df.plot(x='authors', y='citations',
          kind='bar', figsize=(20, 5))

print(df['venue'].value_counts())
print(df['citations'].value_counts())
print(df['fields_of_study'].value_counts())
print(df['topics'].value_counts())

#Assign X as a DataFrame of features and y as a Series of the outcome variable
X = df.drop('citations', 1)
y = df.citations
print(X.head(5))
print("---------------")
print(y.head(5))

#https://www.youtube.com/watch?v=V0u6bxQOUJ8
#models can only handle numeric features, so convert your features into numeric.
#categorical variables > into a set of dummies
#Use get dummmies in Pandas
print(pd.get_dummies(X["is_open_access"]).head(5))

#To be continued for the other features :')

#from sklearn.impute import SimpleImputer
#imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#imp.fit(X)
#X = pd.DataFrame(data = imp.transform(X),columns = X.columns)

#ValueError: Cannot use mean strategy with non-numeric data: could not convert string to float
#Leaving this here just in case.

#Outlier Detection
def find_outliers_tukey(x):
    q1 = np.percentile(x,25)
    q3 = np.percentile(x,75)
    iqr = q3-q1
    floor = q1- 1.5 * iqr
    ceiling = q3 + 1.5 * iqr
    outlier_indices = list(x.index[(x< floor)|(x> ceiling)])
    outlier_values = list(x[outlier_indices])
    
    return outlier_indices, outlier_values

tukey_indices, tukey_values = find_outliers_tukey(X['references'])
print(np.sort(tukey_values))

#Decide which categorical variables you want to use in model
for col_name in X.columns:
    if X[col_name].dtypes =='object':
        unique_cat = len(X[col_name].unique())
        print("Feature '{col_name} has {unique_cat} unique categories".format(
            col_name= col_name, unique_cat = unique_cat))
        
#TypeError: unhashable type: 'list'
#Feature 'doi has 9657 unique categories
#Feature 'title has 9645 unique categories
#Feature 'abstract has 9492 unique categories


#Distribution of Features

#Use pyplot in matplotlib to plot histograms

import matplotlib.pyplot as plt

def plot_histogram(x):
    plt.hist(x, color = 'gray', alpha = 0.5)
    plt.title("Histogram of '{var_name}'".format(var_name = x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    
    
plot_histogram(X['references'])

#Plot histograms to show distribution of features by DV categories.

def plot_histogram_dv(x,y):
    plt.hist(list(x[y==0]), alpha = 0.5, label = 'DV < 100')
    plt.hist(list(x[y==1]), alpha = 0.5, label = 'DV >= 100')
    plt.title("Histogram of '{var_name}' by DV category".format(var_name =x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc = 'upper right')
    plt.show()

#distribution of reference numbers for when citation numbers are less than 100 and >= 100
plot_histogram_dv(X['references'], y)

from scipy.stats import pearsonr
corr = pearsonr(df['references'].fillna(0), df['citations'].fillna(0))
print("r={0}, p={1}".format(*corr))