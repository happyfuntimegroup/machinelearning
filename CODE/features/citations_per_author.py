def citations_per_author(df, author_db):
    """
    !!!! TAKES A LONG TIME --> PARALLELIZATION OR JUST DON'T INTEGRATE?
    Create an array of citations (one number per paper) for every author in a df.
    Used later on to compute the author h-index.
    Input:
        - df:
        - author_db:
    Output:
        - author_db['citations']
    """
    import numpy as np
    import pandas as pd
    #author_db['citations'] = 0
    dic = {}
    for i, i_paper in df.iterrows():
        group = i_paper['authors']
        citations = i_paper['citations']
        for i_author in group:
            condition = author_db['Name'].str.fullmatch(i_author)
            index = author_db[condition].index[0]
            if i_author not in dic:
                dic[i_author] = []
            #author_db.loc[index, 'citations'] += citations
            dic[i_author].append(citations)
    return dic #author_db['citations']