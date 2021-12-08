import pandas as pd
import math

def topic_citations_avarage(data, test):
    """
    Computes the avarage citations for each topic
    Input:
        - df['topics']:    dataframe (dataset); 'topics' column                   [pandas dataframe]
    Output:
        - Avarage citations:                                                            [int]
    """
    citations = [] #create empty list to keep track of citations
    topics_dict = {} #create empty dict to add citations to the topic
    
    out_data = pd.Series(dtype=pd.Float64Dtype())
    out_test = pd.Series(dtype=pd.Float64Dtype())

    for index, i_paper in data.iterrows(): #iterate over dataframe 
        topics = i_paper['topics'] #to get all the topics for one paper
        citation = i_paper['citations'] #and associated citations 

        # iterate over the topics and check if it excists in the dictionary 
        for topic in topics:
            if topic in topics_dict.keys(): # if the topic is in the dict
                citations = topics_dict[topic]
                citations.append(citation) # add citations to list 
                topics_dict[topic] = citations # add list of citations to the topic
            else:
                topics_dict[topic] = [citation] # add topic to the dict
    
    avarage_list = []
    missing_topics = data[data['topics'].str.len() == 0].citations.mean()
    # print(missing_topics)
    for index, i_paper in data.iterrows(): # iterate over the dataframe 
        topics = i_paper['topics'] # check all the topics for one paper
        all_the_citations = [] # create empty list to keep track of all the citations for all the topics
        if len(topics) != 0:
            for topic in topics:
                if topic in topics_dict.keys():
                    all_the_citations += topics_dict[topic] #add citations list of each topic to bigger list
        else:
            all_the_citations = [missing_topics]

        avarage = sum(all_the_citations) / len(all_the_citations) #calculate the avarage of all the citations of each topic
        if avarage is math.nan:
            print('nan')
        out_data[index,] = avarage 
        
    for index, i_paper in test.iterrows(): # iterate over the dataframe 
        topics = i_paper['topics'] # check all the topics for one paper
        all_the_citations = [] # create empty list to keep track of all the citations for all the topics
        if len(topics) != 0:
            for topic in topics:
                if topic in topics_dict.keys():
                    all_the_citations += topics_dict[topic] #add citations list of each topic to bigger list
                else:
                    all_the_citations += [missing_topics]
        else:
            all_the_citations = [missing_topics]

        avarage = sum(all_the_citations) / len(all_the_citations) #calculate the avarage of all the citations of each topic
        if avarage is math.nan:
            print('nan')
        out_test[index,] = avarage 

    return out_data, out_test