from numpy.lib.function_base import average
import pandas as pd
import math
def topic_popularity(data):
    """
    Computes the avarage citations for each topic
    Input:
        - df['topics']:    dataframe (dataset); 'topics' column                   [pandas dataframe]
    Output:
        - Count of field frequency:                                                            [int]
    """
    topic_popularity_dict = {}
    topic_freq = pd.Series(dtype=pd.Int64Dtype())

    for index, i_paper in data.iterrows():

        topics = i_paper['topics']
        
        for topic in topics:
            if topic in topic_popularity_dict.keys():
                topic_popularity_dict[topic] += 1
            else:
                topic_popularity_dict[topic] = 1

    for index, i_paper in data.iterrows():
        topics = i_paper['topics']
        topics_list = []
        for topic in topics:
            topics_list.append(topic_popularity_dict[topic])
        if len(topics_list) != 0:
            most_popular = max(topics_list)
        else:
            most_popular = math.nan
        topic_freq[index,] = most_popular

    return topic_freq
            
            
            
        