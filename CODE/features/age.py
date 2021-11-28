def age(df):
    """
    Computes the age of each paper by taking the year in 'year' column and subtract it from the current year.
    If year is NaN, take the average year of publication for that venue.
    If venue not listed, take the average year of all papers with no venue listed.
    Input:
        - df:   dataframe (dataset)                                         [pandas dataframe]
    Output:
        - age:  vector of age for each paper of the given dataset           [pandas series]
                    with age of each paper                                  [int]
    """
    from datetime import datetime
    import pandas as pd

    # Deal with missing values
    no_year = df[df['year'].isna()] # Return dataframe of all entries with NaN as year of publication
    no_year_venues = [[index, i_paper['venue']] for index, i_paper in no_year.iterrows()]

    mean_year = df[['venue','year']].groupby('venue').mean('year').astype('int') # Get average year of publication for each venue, and turn it into an integer.
    for index, venue in no_year_venues:
        venue_year = mean_year.loc[venue]['year']
        df.loc[index, 'year'] = venue_year
        
    # Publishing year
    publishing_year = df['year']

    # Current year
    current_year = datetime.now().year
    
    # Output
    age = pd.Series(current_year - publishing_year)
    return(age)
