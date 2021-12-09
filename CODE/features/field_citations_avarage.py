import pandas as pd

def field_citations_avarage(data, test):
    """
    For each field found in the TRAIN set, it will make a list of the associated citations. 
    Then, for each paper it checks the fields and combines all the citations lists into a big one. 
    Finally, it will return an avarage of citations based on this big list.  
    Input:
        - df['fields_of_study']:    dataframe (dataset); 'fields_of_study' column                   [pandas dataframe]
    Output:
        - out:      Vector with avarage citations for one paper based on the fields       [pandas series]
    """

    # creating variables for output feature for TEST and TRAIN set 
    citations = [] 
    fields_dict = {} 
    out_data = pd.Series(dtype=pd.Float64Dtype())
    out_test = pd.Series(dtype=pd.Float64Dtype())

    #iterate over dataframe to get all the fields for one paper and associated citations 
    for index, i_paper in data.iterrows(): 
        fields = i_paper['fields_of_study'] 
        citation = i_paper['citations'] 

        # iterate over the topics and check if it excists in the dictionary if the field is in the dict
        for field in fields:
            if field in fields_dict.keys(): 
                citations = fields_dict[field]
                citations.append(citation) 
                fields_dict[field] = citations # add list of citations to the field
            else:
                fields_dict[field] = [citation] # add field to the dict
    
    # if field is missing, it will used the mean of citations based on all the papers with missing topics
    missing_fields = data[data['fields_of_study'].str.len() == 0].citations.mean()

    # check all the topics for one paper and create empty list to keep track of all the citations for all the topics
    for index, i_paper in data.iterrows(): 
        fields = i_paper['fields_of_study'] 
        all_the_citations = [] 
        if len(fields) != 0:
            for field in fields:
                if field in fields_dict.keys():
                    all_the_citations += fields_dict[field] #add citations list of each field to bigger list
        else:
            all_the_citations = [missing_fields]

        #calculate the avarage of all the citations of each field for TRAIN set
        avarage = sum(all_the_citations) / len(all_the_citations) 
        out_data[index,] = avarage 
    
    # check all the topics for one paper in TEST set 
    # and create empty list to keep track of all the citations for all the topics based on TRAIN set dictionary
    for index, i_paper in test.iterrows(): 
        fields = i_paper['fields_of_study'] 
        all_the_citations = [] 
        if len(fields) != 0:
            for field in fields:
                if field in fields_dict.keys():
                    all_the_citations += fields_dict[field] #add citations list of each field to bigger list
                else:
                    all_the_citations += [missing_fields]
        else:
            all_the_citations = [missing_fields]
        
        #calculate the avarage of all the citations of each field for TEST set
        if len(all_the_citations) != 0:
            avarage = sum(all_the_citations) / len(all_the_citations) 
        else:
            avarage = 0
        out_test[index,] = avarage

    return out_data, out_test