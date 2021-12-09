import pandas as pd

def topics_variety(data):
    """
    Computes the number of different topics that each paper is about by taking the number of topics in 'topics'
    Input:
        - df['topics']:    dataframe (dataset); 'topics' column                           [pandas dataframe]
    Output:
        - Topic variety:   vector of topic_variety for each paper of the given dataset      [pandas series]
                            with Topic_variety                                                   [int]
    """

    # Calculates how many topics a paper has
    Topic_variety = pd.Series([len(i) for i in data['topics']]) 

    return Topic_variety                                           