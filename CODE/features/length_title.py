def length_title(data):
    """
    Computes the length of the title from a paper by counting how many words it contains.
    This function is based on an online published article:
        https://www.natureindex.com/news-blog/five-features-highly-cited-scientific-article
    Input:
        - df['title']:    dataframe (dataset); 'title' column                           [pandas dataframe]
    Output:
        - Title_length:   vector of wordlength of for each paper title of the given dataset      [pandas series]
                            with Title_length                                                   [int]
    """
    import pandas as pd
    
    out = pd.Series(dtype=pd.Float64Dtype())

    for index, i_paper in data.iterrows():
        title = i_paper['title']
        Title_length = len(title.split())
        out[index,] = Title_length

    return out

