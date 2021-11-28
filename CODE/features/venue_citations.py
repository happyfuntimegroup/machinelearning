import pandas as pd

def venues_citations(data, venue_db):
    """Output: 
    {Venue : [list of citations per paper]
    """
    venues = {}
    citations = []
    venue_venPresL = {}
    
    out = pd.Series(dtype=pd.Float64Dtype())
    
    for index, i_paper in data.iterrows():
        venue = i_paper['venue']
    
        if venue in venues.keys():
            citations = venues[venue]
            citations.append(i_paper['citations'])
            venues[venue] = citations
        else:
            citations.append(i_paper['citations'])
            venues[venue] = [(i_paper['citations'])]


    for key, value in venues.items():
        sum_venue = sum(value)
        venPresL = (sum_venue / len(value))      
        venue_venPresL[key] = venPresL
    
    # For each paper in the dataset, add the venPresL score to that paper; i.e. look at the venue in which it was published,
    #   and add the score to that index in the pandas Series.
    for index, i_paper in data.iterrows():
        venue = i_paper['venue']
        out[index,] = venue_venPresL[venue]
    
    # Output
    return out # Pandas Series of VenPresL for each paper