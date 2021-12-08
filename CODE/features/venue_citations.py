import pandas as pd

def venues_citations(data, test):
    """Output: 
    {Venue : [list of citations per paper]
    """
    venues = {}
    citations = []
    venue_venPresL = {}
    
    out_data = pd.Series(dtype=pd.Float64Dtype())
    out_test = pd.Series(dtype=pd.Float64Dtype())
    
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
    
    missing_venue = sum(venue_venPresL.values())/len(venue_venPresL.values())
    # For each paper in the dataset, add the venPresL score to that paper; i.e. look at the venue in which it was published,
    #   and add the score to that index in the pandas Series.
    for index, i_paper in data.iterrows():
        venue = i_paper['venue']
        out_data[index,] = venue_venPresL[venue]
        
    for index, i_paper in test.iterrows():
        venue = i_paper['venue']
        if venue in venue_venPresL.keys():
            out_test[index,] = venue_venPresL[venue]
        else:
            out_test[index,] = missing_venue
    
    # Output
    return out_data, out_test # Pandas Series of VenPresL for each paper