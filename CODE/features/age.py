def age(df):
    """
    Computes the age of each paper by taking the year in 'year' column and subtract it from the current year
    Input:
        - df['year']:   dataframe (dataset); or just the 'year' column              [pandas dataframe or series]
    Output:
        - Age:          vector of age for each paper of the given dataset           [pandas series]
                            with age of each paper                                  [int]
    """
    from datetime import datetime
    import pandas as pd
    
    publishing_year = df['year']
    current_year = datetime.now().year
    
    Age = pd.Series(current_year - publishing_year)
    return(Age)
