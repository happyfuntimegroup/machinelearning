
def missing_values1 (data):
    """
    Deals in place with missing values in specified columns: 
    'fields_of_study', 'title', 'abstract', 'authors', 'venue', 'references', 'topics'
    """
    data.loc[data['fields_of_study'].isnull(), 'fields_of_study'] = ""
    data.loc[data['title'].isnull(), 'title'] = ""
    data.loc[data['abstract'].isnull(), 'abstract'] = ""
    data.loc[data['authors'].isnull(), 'authors'] = ""
    data.loc[data['venue'].isnull(), 'venue'] = ""
    data.loc[data['references'].isnull(), 'references'] = ""
    data.loc[data['topics'].isnull(), 'topics'] = ""


def missing_values2 (data):
    """
    Deals in place with missing values in specified columns: 'open_access' and 'year'
    """
    # Open_access, thanks to jreback (27th of July 2016) https://github.com/pandas-dev/pandas/issues/13809
    OpAc_by_venue = data.groupby('venue').open_access.apply(lambda x: x.mode()) # Take mode for each venue
    OpAc_by_venue = OpAc_by_venue.to_dict()
    missing_OpAc = data.loc[data['open_access'].isnull(),]
    for i, i_paper in missing_OpAc.iterrows():
        venue = i_paper['venue']
        doi = i_paper['doi']
        index = data[data['doi'] == doi].index[0]
        if venue in OpAc_by_venue.keys():   # If a known venue, append the most frequent value for that venue
            data.loc[index,'open_access'] = OpAc_by_venue[venue] # Set most frequent occurrence 
        else:                               # Else take most occurring value in entire dataset
            data.loc[index,'open_access'] = data.open_access.mode()[0] # Thanks to BENY (2nd of February, 2018) https://stackoverflow.com/questions/48590268/pandas-get-the-most-frequent-values-of-a-column

    # Year
    year_by_venue = data.groupby('venue').year.apply(lambda x: x.mean()) # Take mean for each venue
    year_by_venue = year_by_venue.to_dict()
    missing_year = data.loc[data['year'].isnull(),]
    for i, i_paper in missing_year.iterrows():
        venue = i_paper['venue']
        doi = i_paper['doi']
        index = data[data['doi'] == doi].index[0]
        if venue in year_by_venue.keys():   # If a known venue, append the mean value for that venue
            data.loc[index, 'year'] = year_by_venue[venue] # Set mean publication year
        else:                               # Else take mean value of entire dataset
            data.loc[index,'year'] = data.year.mean()
        
