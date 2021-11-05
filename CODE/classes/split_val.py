from sklearn.model_selection import train_test_split 
X = train.loc[:, train.columns != 'citations']
y = train.loc[:, train.columns == 'citations']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)