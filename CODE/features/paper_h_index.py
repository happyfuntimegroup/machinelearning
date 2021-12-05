def paper_h_index(data, author_citation_dic):
    """
    Returns one specific h_index for each paper, as there may be multiple authors.
        First, make a list with the h_index for each author for that paper. 
        Then return a specific h_index measure: here we use the highest h_index that occurs for that group of authors.
    Input:
        - data:                 dataframe/dataset                                               [pandas dataframe]
        - author_citation_dic:  dataset (dict) of all authors (keys) in the known dataset 
                                 and a list with citations of papers they worked on (values)    [dict]
    Output:
        - out:                  vector with specific h_index for each paper                     [pandas series]
    """
    import pandas as pd
    import math
    from CODE.features.author_h_index import author_h_index
    
    out = pd.Series(dtype=pd.Float64Dtype())
    h_index = author_h_index(author_citation_dic)
    for index, i_paper in data.iterrows():
        h_team = {}
        authors = i_paper['authors']
        for i_author in authors:
            h_team[i_author] = h_index[i_author]
        if len(h_team) != 0:
            max_h = max(h_team.values())
        else:
            max_h = math.nan
        out.loc[index] = max_h
    return(out)