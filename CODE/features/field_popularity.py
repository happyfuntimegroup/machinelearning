import pandas as pd
import math

def field_popularity(data, test):
    """
    Computes the number of different fields that each paper is about by taking the number of fields in 'fields_of_study'
    Input:
        - df['fields_of_study']:    dataframe (dataset); 'fields_of_study' column                   [pandas dataframe]
    Output:
        - field_freq:           Vector of count of the most popular field      [pandas series]
                                   
    """
    
    # creating variables for output feature for TEST and TRAIN set 
    field_popularity_dict = {}
    field_freq_data = pd.Series(dtype=pd.Int64Dtype())
    field_freq_test = pd.Series(dtype=pd.Int64Dtype())

    # check TRAIN data for fields, and counts the frequency
    for index, i_paper in data.iterrows():
        fields = i_paper['fields_of_study']

        for field in fields:
            if field in field_popularity_dict.keys():
                field_popularity_dict[field] += 1
            else:
                field_popularity_dict[field] = 1

    # if fields are missing it returns the avarage frequency           
    missing_fields = sum(field_popularity_dict.values())/len(field_popularity_dict.values())

    # checks TRAIN set and returns the count of most popular field
    for index, i_paper in data.iterrows():
        fields = i_paper['fields_of_study']
        field_list = []
        for field in fields:
            field_list.append(field_popularity_dict[field])
        if len(field_list) != 0:
            most_popular = max(field_list)
        else:
            most_popular = int(missing_fields)
        field_freq_data[index,] = most_popular

    # checks TEST set and returns the count of most popular field based on field count in TRAIN set
    for index, i_paper in test.iterrows():
        fields = i_paper['fields_of_study']
        field_list = []
        for field in fields:
            if field in field_popularity_dict.keys():
                field_list.append(field_popularity_dict[field])
        if len(field_list) != 0:
            most_popular = max(field_list)
        else:
            most_popular = int(missing_fields)
        field_freq_test[index,] = most_popular

    return field_freq_data, field_freq_test
        



