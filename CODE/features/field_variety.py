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

    field_popularity_dict = {}
    fields_dict = {}


    for index, i_paper in data.iterrows():
        citations = []

        fields = i_paper['fields_of_study']
        if fields == None:
            fields = ['Missing']

        citation = i_paper['citations']

        for field in fields:
            if field in field_popularity_dict.keys():
                field_popularity_dict[field] += 1
            else:
                field_popularity_dict[field] = 1
            
            if field in fields_dict.keys():
                citations.append(citation)
                fields_dict[field] = sum(citations) / len(citations)
            else:
                fields_dict[field] = citation

        return Field_variety, field_popularity_dict
        



