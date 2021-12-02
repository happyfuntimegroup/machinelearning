import pandas as pd

def venue_popularity(data):
    venues_popularity_dict = {}
    
    venue_reformatted = pd.Series()
    
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
    
        venue_reformatted[index,] = venue
    
    # Check venue and add venue_frequency to each paper
    venue_freq = pd.Series(dtype=pd.Int64Dtype())
    for index, i_paper in data.iterrows():
        venue_freq[index,] = venues_popularity_dict[venue_reformatted[index]] 

    return venue_freq, venue_reformatted
