import pandas as pd

def venues_citations(source_file):
    """Output: 
    {Venue : [list of citations per paper]
    """
    papers = pd.read_json(source_file)
    venues = {}

    for i in range(len(papers)):
        venue_temp = papers.iloc[i]['venue']
        venue = ''

        for char in venue_temp:
            if char.isupper():
                venue += char

            if venue in venues.keys():
                citations = venues[venue]
                citations.append(papers.iloc[i]['citations'])
                venues[venue] = citations
            else:
                citations.append(papers.iloc[i]['citations'])
                venues[venue] = [(papers.iloc[i]['citations'])]


        for key, value in venues.items():
            print(key, ':', value)
            sum_venue = sum(value)
            print(key, ':', sum_venue)
            venPresL = (sum_venue / len(value))
            print(venPresL)
            


