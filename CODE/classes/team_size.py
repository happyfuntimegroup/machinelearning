def team_size(df):
    """
    Computes team size of each paper by taking the number of authors in 'authors'
    Input:
        - df['authors']:    dataframe (dataset); 'authors' column       [pandas dataframe]
    Output:
        - team_size:        vector of team_size for each paper of the given dataset         [pandas series]
                            with team_size                                                  [int]
    """"
    import pandas as pd

    team_size = pd.Series([len(i) for i in df['authors']])      # Teamsize

    return(team_size)                                           # Output

    