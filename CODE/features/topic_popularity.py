from numpy.lib.function_base import average
import pandas as pd

def topic_popularity(data):
    """
    Computes the avarage citations for each topic
    Input:
        - df['topics']:    dataframe (dataset); 'topics' column                   [pandas dataframe]
    Output:
        - Count of field frequency:                                                            [int]
    """
    topic_popularity_dict = {}

    for index, i_paper in data.iterrows():

        topics = i_paper['topics']
        citation = i_paper['citations']
        
        for topic in topics:
            if topic in topic_popularity_dict.keys():
                topic_popularity_dict[topic] += 1
            else:
                topic_popularity_dict[topic] = 1

    return topic_popularity_dict
            
            
            
        