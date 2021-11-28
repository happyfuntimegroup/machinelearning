import pandas as pd

def venue_frequency(data):
    papers = data
    venues_count = {}
    count = 1
    
    venue_reformatted = pd.Series()
    
    for index, i_paper in data.iterrows():
        venue_temp = papers.iloc[index]['venue']
        venue = ''

        for char in venue_temp:
            if char.isupper():
                venue += char
            
        if venue in venues_count.keys():
            venues_count[venue] += 1
        else:
            venues_count[venue] = 1
    
        venue_reformatted[index,] = venue
    return venues_count, venue_reformatted
