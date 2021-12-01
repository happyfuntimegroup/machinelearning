import pandas as pd

#The number of different fields that a certain article is about > 1 or 2 or 3 

def field_variety(data):
    """
    Computes the number of different fields that each paper is about by taking the number of fields in 'fields_of_study'
    Input:
        - df['fields_of_study']:    dataframe (dataset); 'fields_of_study' column                   [pandas dataframe]
    Output:
        - Field variety:           vector of field_variety for each paper of the given dataset      [pandas series]
                                    with field_variety                                                   [int]
    """
    
    Field_variety = pd.Series([len(i) for i in data['fields_of_study']])      # Variety of fields
    
    return Field_variety
        



