import pandas as pd

def topic_popularity(data, test):
    """
    Compute popularity for each topic and returns for each paper in the dataset the most popular (highest) topic frequency (integer).
    
    Input:
        - df['topics']:     dataframe (dataset)                   [pandas dataframe]
    Output:
        - topic_freq:       Vector with most popular topic frequency for each paper.   [pandas series of integers]                                                         [int]
    """
    
    # creating variables for output feature for TEST and TRAIN set 
    topic_popularity_dict = {}
    topic_freq_data = pd.Series(dtype=pd.Int64Dtype())
    topic_freq_test = pd.Series(dtype=pd.Int64Dtype())

    # check TRAIN data for topics, and counts the frequency
    for index, i_paper in data.iterrows():
        topics = i_paper['topics']
        for topic in topics:
            if topic in topic_popularity_dict.keys():
                topic_popularity_dict[topic] += 1
            else:
                topic_popularity_dict[topic] = 1

    # if topics are missing it returns the avarage frequency    
    missing_topics = sum(topic_popularity_dict.values())/len(topic_popularity_dict.values())

    # checks TRAIN set and returns the count of most popular topic
    for index, i_paper in data.iterrows():
        topics = i_paper['topics']
        topics_list = []
        for topic in topics:
            topics_list.append(topic_popularity_dict[topic])
        if len(topics_list) != 0:
            most_popular = max(topics_list)
        else:
            most_popular = int(missing_topics)
        topic_freq_data[index,] = most_popular
    
    # checks TEST set and returns the count of most popular topic based on field count in TRAIN set
    for index, i_paper in test.iterrows():
        topics = i_paper['topics']
        topics_list = []
        for topic in topics:
            if topic in topic_popularity_dict.keys():
                topics_list.append(topic_popularity_dict[topic])
        if len(topics_list) != 0:
            most_popular = max(topics_list)
        else:
            most_popular = int(missing_topics)
        topic_freq_test[index,] = most_popular

    return topic_freq_data, topic_freq_test
            
            
            
        