import pandas as pd

def topic_citations_avarage(data):
    """
    Computes the avarage citations for each topic
    Input:
        - df['topics']:    dataframe (dataset); 'topics' column                   [pandas dataframe]
    Output:
        - Avarage citations:                                                            [int]
    """
    citations = []
    topics_dict = {}
    
    out = pd.Series(dtype=pd.Float64Dtype())

    for index, i_paper in data.iterrows():

        topics = i_paper['topics']
        citation = i_paper['citations']

        for topic in topics:
            if topic in topics_dict.keys():
                citations.append(citation)
                topics_dict[topic] = citations
            else:
                topics_dict[topic] = citation

    for index, i_paper in data.iterrows():
        topics = i_paper['topics']
        all_the_citations = []
        for topic in topics:
            if topic in topics_dict.keys():
                all_the_citations.append(topics_dict[topic])
        
        avarage = sum(all_the_citations) / len(all_the_citations)
        out[index,] = avarage 