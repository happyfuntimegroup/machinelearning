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
    
   # field_variety = pd.Series([len(i) for i in fields_filled])            # Variety of fields
    # field_variety = pd.Series([len(i) for i in data['fields_of_study']])      # Variety of fields

    field_variety = 0
    field_popularity_dict = {}

    # Support: Fraction of transactions that contain an itemset.
    # support(field) = number of papers with that field / total number of papers 

    # Confidence: Measures how often items in Y appear in transactions that contain X

    for index, i_paper in data.iterrows():

        fields = i_paper['fields_of_study']
        if fields == None:
            fields = ['Missing']

        field_variety = len(fields)

        for field in fields:
            if field in field_popularity_dict.keys():
                field_popularity_dict[field] += 1
            else:
                field_popularity_dict[field] = 1
    
    print(field_popularity_dict)

        
        # df_fields['Index'] = index
        # # df_fields['Fields'] = fields
        # df_fields['Field Variety'] = field_variety
        # df_fields['Popularity'] = 

        # print(df_fields)

        



