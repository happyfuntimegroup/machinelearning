import pandas as pd
def length_title(data):
    """
    Computes the length of the title from a paper by counting how many words it contains.
    This function is based on an online published article:
        https://www.natureindex.com/news-blog/five-features-highly-cited-scientific-article
    Input:
        - df['title']:    dataframe (dataset); 'title' column                                   [pandas dataframe]
    Output:
        - title_length:   vector of wordlength of for each paper title of the given dataset     [pandas series]
                            with title_length                                                   [int]
    """    
    out = pd.Series(dtype=pd.Float64Dtype())

    for index, i_paper in data.iterrows():
        title = i_paper['title']
        title_length = len(title.split())
        out[index,] = title_length          # Add title length to output for each paper

    # Output
    return out

