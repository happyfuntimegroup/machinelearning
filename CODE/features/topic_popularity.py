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
        
        for topic in topics:
            if topic in topic_popularity_dict.keys():
                topic_popularity_dict[topic] += 1
            else:
                topic_popularity_dict[topic] = 1

    topic_freq = pd.Series(dtype=pd.Int64Dtype())
    for index, i_paper in data.iterrows():
        topic_freq[index,] = topic_popularity_dict[i_paper['topics']] 

    return topic_freq
            
            
            
        