import pandas as pd
import math
def field_popularity(data):
    """
    Computes the number of different fields that each paper is about by taking the number of fields in 'fields_of_study'
    Input:
        - df['fields_of_study']:    dataframe (dataset); 'fields_of_study' column                   [pandas dataframe]
    Output:
        - Field variety:           vector of field_variety for each paper of the given dataset      [pandas series]
                                    with field_variety                                                   [int]
    """
    
    field_popularity_dict = {}
    field_freq = pd.Series(dtype=pd.Int64Dtype())


    for index, i_paper in data.iterrows():
        fields = i_paper['fields_of_study']

        for field in fields:
            if field in field_popularity_dict.keys():
                field_popularity_dict[field] += 1
            else:
                field_popularity_dict[field] = 1
    
    for index, i_paper in data.iterrows():
        fields = i_paper['fields_of_study']
        field_list = []
        for field in fields:
            field_list.append(field_popularity_dict[field])
        if len(field_list) != 0:
            most_popular = max(field_list)
        else:
            most_popular = math.nan
        field_freq[index,] = most_popular

    return field_freq
        



