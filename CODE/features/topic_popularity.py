from numpy.lib.function_base import average
import pandas as pd

def topic_popularity(data):
    """ returns dictionary key = topic, value = average citions for that topic """

    topic_popularity_dict = {}
    citations = []
    topics_dict = {}

    for index, i_paper in data.iterrows():

        topics = i_paper['topics']
        if topics == None:
            topics = ['Missing']
        
        citation = i_paper['citations']
        
        for topic in topics:
            if topic in topic_popularity_dict.keys():
                topic_popularity_dict[topic] += 1
            else:
                topic_popularity_dict[topic] = 1

            if topic in topics_dict.keys():
                citations.append(citation)
                topics_dict[topic] = sum(citations) / len(citations)
            else:
                topics_dict[topic] = citation

    print(topic_popularity_dict)
    print(topics_dict)
            
            
            
        