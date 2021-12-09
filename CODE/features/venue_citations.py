import pandas as pd

def venues_citations(data, test):
    """
    For each venue found in the TRAIN set, it will make a list of the associated citations. 
    Then, it will return an avarage of citations based on this list (venPresL). 
    This feature is based on methods of a previous study:  
        Chakraborty, T., Kumar, S., Goyal, P., Ganguly, N., & Mukherjee, A., (2014). Towards a Stratified Learning Approach 
        to Predict Future Citation Counts, IEEE/ACM Joint Conference on Digital Libraries, pp. 351 - 360. doi: 10.1109/JCDL.2014.6970190
    Input:
        - df['venue']:    dataframe (dataset); 'venue' column                   [pandas dataframe]
    Output:
        - out:      Vector with avarage citations for one paper based on the venue      [pandas series]
    """

    # creating variables for output feature for TEST and TRAIN set 
    venues = {}
    citations = []
    venue_venPresL = {}
    out_data = pd.Series(dtype=pd.Float64Dtype())
    out_test = pd.Series(dtype=pd.Float64Dtype())
    
    # iterates over TRAIN set and makes a dictionary to keep track of venues and associated citations
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
    
    # calculates an avarage for the papers that have no venue variable 
    missing_venue = sum(venue_venPresL.values())/len(venue_venPresL.values())

    # for each paper in the dataset, add the venPresL score to that paper; i.e. look at the venue in which it was published,
    #   and add the score to that index in the pandas Series.
    for index, i_paper in data.iterrows():
        venue = i_paper['venue']
        out_data[index,] = venue_venPresL[venue]
        
    # checks venues of TEST set and returns venPresL based on TRAIN set 
    for index, i_paper in test.iterrows():
        venue = i_paper['venue']
        if venue in venue_venPresL.keys():
            out_test[index,] = venue_venPresL[venue]
        else:
            out_test[index,] = missing_venue

    return out_data, out_test 