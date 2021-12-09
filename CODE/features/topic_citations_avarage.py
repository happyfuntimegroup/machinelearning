import pandas as pd

def topic_citations_avarage(data, test):
    """
    For each topic found in the TRAIN set, it will make a list of the associated citations. 
    Then, for each paper it checks the topics and combines all the citations lists into a big one. 
    Finally, it will return an avarage of citations based on this big list.  
    Input:
        - df['topics']:    dataframe (dataset); 'topics' column                   [pandas dataframe]
    Output:
        - out:      Vector with avarage citations for one paper based on the topics       [pandas series]
    """

    # creating variables for output feature for TEST and TRAIN set 
    citations = [] 
    topics_dict = {} 
    out_data = pd.Series(dtype=pd.Float64Dtype())
    out_test = pd.Series(dtype=pd.Float64Dtype())

    #iterate over dataframe to get all the topics for one paper and associated citations 
    for index, i_paper in data.iterrows(): 
        topics = i_paper['topics'] 
        citation = i_paper['citations'] 

        # iterate over the topics and check if it excists in the dictionary if the topic is in the dict
        for topic in topics:
            if topic in topics_dict.keys(): 
                citations = topics_dict[topic]
                citations.append(citation) 
                topics_dict[topic] = citations # add list of citations to the topic
            else:
                topics_dict[topic] = [citation] # add topic to the dict

    # if a topic is missing, it will used the mean of citations based on all the papers with missing topics
    missing_topics = data[data['topics'].str.len() == 0].citations.mean()

    # check all the topics for one paper and create empty list to keep track of all the citations for all the topics
    for index, i_paper in data.iterrows(): 
        topics = i_paper['topics'] 
        all_the_citations = [] 
        if len(topics) != 0:
            for topic in topics:
                if topic in topics_dict.keys():
                    all_the_citations += topics_dict[topic] #add citations list of each topic to bigger list
        else:
            all_the_citations = [missing_topics]

        #calculate the avarage of all the citations of each topic for TRAIN set
        avarage = sum(all_the_citations) / len(all_the_citations) 
        out_data[index,] = avarage 

    # check all the topics for one paper in TEST set 
    # and create empty list to keep track of all the citations for all the topics based on TRAIN set dictionary
    for index, i_paper in test.iterrows(): 
        topics = i_paper['topics'] 
        all_the_citations = []
        if len(topics) != 0:
            for topic in topics:
                if topic in topics_dict.keys():
                    all_the_citations += topics_dict[topic] 
                else:
                    all_the_citations += [missing_topics]
        else:
            all_the_citations = [missing_topics]

        #calculate the avarage of all the citations of each topic for TEST set
        avarage = sum(all_the_citations) / len(all_the_citations) 
        out_test[index,] = avarage 

    return out_data, out_test