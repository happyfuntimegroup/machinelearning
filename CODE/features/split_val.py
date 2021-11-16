def split_val(df, target_variable = 'citations', val_size = 0.33, random = 42):
    """
    Function to split dataset into train and validation set.
    Input:
        df:                 dataset                                                         [Pandas dataframe]
        target_variable:    target variable that will be used to split the dataset into 
                            sets containing features and target                             [string; default = 'citations']
        val_size:           proportion of the dataset that should be made into 
                            validation set                                                  [number between 0-1; default = 0.33]
        random:             set a seed for reproducibility                                  [default = 42]
    Output:
        X_train:        training set, excluding the set of target values (citations)        [Pandas dataframe]
        X_val:          validation set, excluding the set of target values (citations)      [Pandas dataframe]
        y_train:        target values for the training set (citations)                      [Pandas dataframe]
        y_val:          target values for the validation set (citations)                    [Pandas dataframe]
    """
    from sklearn.model_selection import train_test_split 
    X = df.loc[:, df.columns != target_variable]
    y = df.loc[:, df.columns == target_variable]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_size, random_state = random)
    
    return X_train, X_val, y_train, y_val