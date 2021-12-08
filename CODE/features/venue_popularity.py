import pandas as pd

def venue_popularity(data, test):

    venues_popularity_dict = {}
    venue_reformatted_data = pd.Series()
    venue_reformatted_test = pd.Series()
    
    for index, i_paper in data.iterrows():
        venue_temp = i_paper['venue']
        venue = ''

        for char in venue_temp:
            if char.isupper():
                venue += char
            
        if venue in venues_popularity_dict.keys():
            venues_popularity_dict[venue] += 1
        else:
            venues_popularity_dict[venue] = 1
    
        venue_reformatted_data[index,] = venue
    
    # check test data for new venues, and add them to the reformatted venues 
    for index, i_paper in test.iterrows():
        venue_temp = i_paper['venue']
        venue = ''

        for char in venue_temp:
            if char.isupper():
                venue += char
    
        venue_reformatted_test[index,] = venue
    
    # Check venue and add venue_frequency to each paper
    venue_freq_data = pd.Series(dtype=pd.Int64Dtype())
    venue_freq_test = pd.Series(dtype=pd.Int64Dtype())
    for index, i_paper in data.iterrows():
        venue_freq_data[index,] = venues_popularity_dict[venue_reformatted_test[index]] 
    
    for index, i_paper in test.iterrows():
        venue_freq_test[index,] = venues_popularity_dict[venue_reformatted_test[index]] 

    return venue_freq_data, venue_reformatted_data, venue_freq_test, venue_reformatted_test
