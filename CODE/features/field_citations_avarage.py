import pandas as pd
import math

def field_citations_avarage(data, test):
    """
    Computes the avarage citations for each field
    Input:
        - df['fields_of_study']:    dataframe (dataset); 'fields_of_study' column                   [pandas dataframe]
    Output:
        - Avarage citations:                                                            [int]
    """
    citations = [] #create empty list to keep track of citations
    fields_dict = {} #create empty dict to add citations to the field
    
    out_data = pd.Series(dtype=pd.Float64Dtype())
    out_test = pd.Series(dtype=pd.Float64Dtype())

    for index, i_paper in data.iterrows(): #iterate over dataframe 
        fields = i_paper['fields_of_study'] #to get all the fields for one paper
        citation = i_paper['citations'] #and associated citations 

        # iterate over the topics and check if it excists in the dictionary 
        for field in fields:
            if field in fields_dict.keys(): # if the topic is in the dict
                citations = fields_dict[field]
                citations.append(citation) # add citations to list 
                fields_dict[field] = citations # add list of citations to the field
            else:
                fields_dict[field] = [citation] # add topic to the dict
    
    missing_fields = data[data['fields_of_study'].str.len() == 0].citations.mean()

    for index, i_paper in data.iterrows(): # iterate over the dataframe 
        fields = i_paper['fields_of_study'] # check all the topics for one paper
        all_the_citations = [] # create empty list to keep track of all the citations for all the fields
        if len(fields) != 0:
            for field in fields:
                if field in fields_dict.keys():
                    all_the_citations += fields_dict[field] #add citations list of each field to bigger list
        else:
            all_the_citations = [missing_fields]

        print(all_the_citations)
        # avarage = sum(all_the_citations) / len(all_the_citations) #calculate the avarage of all the citations of each field
        # out_data[index,] = avarage 
    
    # for index, i_paper in test.iterrows(): # iterate over the dataframe 
    #     fields = i_paper['fields_of_study'] # check all the topics for one paper
    #     all_the_citations = [] # create empty list to keep track of all the citations for all the fields
    #     if len(fields) != 0:
    #         for field in fields:
    #             if field in fields_dict.keys():
    #                 all_the_citations += fields_dict[field] #add citations list of each field to bigger list
    #             else:
    #                 all_the_citations += missing_fields
    #     else:
    #         all_the_citations = [missing_fields]
        
    #     print(all_the_citations)

    #     if len(all_the_citations) > 0:
    #         print(all_the_citations)
    #         print(len(all_the_citations))
    #         # avarage = sum(all_the_citations) / len(all_the_citations) #calculate the avarage of all the citations of each topic
    #     else:
    #         avarage = 0
        # out_test[index,] = avarage

    return out_data, out_test

