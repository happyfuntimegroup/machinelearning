import pandas as pd

def venue_frequency(source_file):
    papers = pd.read_json(source_file)
    venues_count = {}
    count = 1

    for i in range(len(papers)):
        venue_temp = papers.iloc[i]['venue']
        venue = ''

        for char in venue_temp:
            if char.isupper():
                venue += char
            
            if venue in venues_count.keys():
                venues_count[venue] += 1
            else:
                venues_count[venue] = 1
    
    return venues_count
