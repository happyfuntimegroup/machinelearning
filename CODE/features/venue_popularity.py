import pandas as pd

def venue_popularity(data, test):
    """
    Returns the frequency of a venue based on the count in de database and also returns the venues reformatted.  
        First, it get the venue and reformats the string by only saving the capital letters. This needs to be done
        so all the venues have the same format. 
        Then is iterates over the database again, to count the frequency (popularity) of that venue.
    Input:
        - df['venue']:     dataframe (dataset)                   [pandas dataframe]      
        - venue:  dataset (dict) of all authors (keys) in the known dataset 
    Output:
        - venue_freq_data:   Vector with most popular topic frequency for each paper.          [pandas series]
        - venue_reformatted: Vector with reformatted strings for the venue variable            [pandas series]
    """

    # creating variables for output feature for TEST and TRAIN set 
    venues_popularity_dict = {}
    venue_reformatted_data = pd.Series()
    venue_reformatted_test = pd.Series()
    venue_freq_data = pd.Series(dtype=pd.Int64Dtype())
    venue_freq_test = pd.Series(dtype=pd.Int64Dtype())

    # check TRAIN data for new venues, and add them to the reformatted venues
    for index, i_paper in data.iterrows():
        venue_temp = i_paper['venue']
        venue = ''
        for char in venue_temp:
            if char.isupper():
                venue += char

        # check venue and add venue count to each paper
        if venue in venues_popularity_dict.keys():
            venues_popularity_dict[venue] += 1
        else:
            venues_popularity_dict[venue] = 1
    
        venue_reformatted_data[index,] = venue
    
    # check TEST data for new venues, and add them to the reformatted venues 
    for index, i_paper in test.iterrows():
        venue_temp = i_paper['venue']
        venue = ''
        for char in venue_temp:
            if char.isupper():
                venue += char

        # check venue and add venue count to each paper
        if venue in venues_popularity_dict.keys():
            venues_popularity_dict[venue] += 1
        else:
            venues_popularity_dict[venue] = 1
    
        venue_reformatted_test[index,] = venue
    

    for index, i_paper in data.iterrows():
        venue_freq_data[index,] = venues_popularity_dict[venue_reformatted_data[index]] 
    
    for index, i_paper in test.iterrows():
        venue_freq_test[index,] = venues_popularity_dict[venue_reformatted_test[index]] 

    return venue_freq_data, venue_reformatted_data, venue_freq_test, venue_reformatted_test
