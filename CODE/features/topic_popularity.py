from numpy.lib.function_base import average
import pandas as pd
def topic_popularity(data):
    """
    Compute popularity for each topic and returns for each paper in the dataset the most popular (highest) topic frequency (integer).
    
    Input:
        - df['topics']:     dataframe (dataset)                   [pandas dataframe]
    Output:
        - topic_freq:       Vector with most popular topic frequency for each paper.   [pandas series of integers]                                                         [int]
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

    missing_topics = sum(topic_popularity_dict.values())/len(topic_popularity_dict.values())

    for index, i_paper in data.iterrows():
        topics = i_paper['topics']
        topics_list = []
        for topic in topics:
            topics_list.append(topic_popularity_dict[topic])
        if len(topics_list) != 0:
            most_popular = max(topics_list)
        else:
            most_popular = int(missing_topics)
        topic_freq[index,] = most_popular

    return topic_freq
            
            
            
        