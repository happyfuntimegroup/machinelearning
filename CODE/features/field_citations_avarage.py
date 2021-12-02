import pandas as pd

def field_citations_avarage(data):
    """
    Computes the avarage citations for each topic
    Input:
        - df['fields_of_study']:    dataframe (dataset); 'fields_of_study' column                   [pandas dataframe]
    Output:
        - Avarage citations:                                                            [int]
    """
    citations = []
    fields_dict = {}
    
    out = pd.Series(dtype=pd.Float64Dtype())

    for index, i_paper in data.iterrows():

        fields = i_paper['fields_of_study']
        citation = i_paper['citations']

        for field in fields:
            if field in fields_dict.keys():
                citations.append(citation)
                fields_dict[field] = citations
            else:
                fields_dict[field] = citation

    for index, i_paper in data.iterrows():
        fields = i_paper['fields_of_study']
        all_the_citations = []
        for field in fields:
            if field in fields_dict.keys():
                all_the_citations.append(fields_dict[field])
        
        avarage = sum(all_the_citations) / len(all_the_citations)
        out[index,] = avarage 
    
    return out 