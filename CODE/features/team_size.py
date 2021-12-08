import pandas as pd
def team_size(df):
    """
    Computes team size of each paper by taking the number of authors in 'authors'
    Input:
        - df:               dataframe (dataset); or just the 'authors' column               [pandas dataframe]
    Output:
        - team:             vector of team_size for each paper of the given dataset         [pandas series]
                            with team_size                                                  [int]
    """
    team = pd.Series([len(i) for i in df['authors']])      # teamsize

    # Output
    return(team)