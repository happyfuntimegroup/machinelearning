def paper_h_index(data, author_citation_dic, test):
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
    from CODE.features.author_h_index import author_h_index
    
    out_data = pd.Series(dtype=pd.Float64Dtype())
    out_test = pd.Series(dtype=pd.Float64Dtype())
    h_index = author_h_index(author_citation_dic)
    missing_h = sum(h_index.values())/len(h_index.values())
    
    for index, i_paper in data.iterrows():
        h_team = {}
        authors = i_paper['authors']
        for i_author in authors:
            h_team[i_author] = h_index[i_author]
        if len(h_team) != 0:
            max_h = max(h_team.values())
        else:
            max_h = missing_h
        out_data.loc[index] = max_h

    for index, i_paper in test.iterrows():
        h_team = {}
        authors = i_paper['authors']
        for i_author in authors:
            if i_author in h_index.keys():
                h_team[i_author] = h_index[i_author]
            else:
                h_team[i_author] = missing_h
        if len(h_team) != 0:
            max_h = max(h_team.values())
        else:
            max_h = missing_h
        out_test.loc[index] = max_h

    return out_data, out_test