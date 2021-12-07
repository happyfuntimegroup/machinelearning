"""
input: data, column, upper threshold
"""

def outlier_threshold (data, col, threshold):
    a = data[data[col] > threshold]
    
    return a


# X_train = X_train[X_train['references'] < 500]
# X_train = X_train[X_train['team_sz'] < 40]
# X_train = X_train[X_train['topic_var'] < 60]
# X_train = X_train[X_train['venPresL'] < 300]
# X_train = X_train[X_train['h_index'] < 30]

# y_train = y_train[y_train['references'] < 500]
# y_train = y_train[y_train['team_sz'] < 40]
# y_train = y_train[y_train['topic_var'] < 60]
# y_train = y_train[y_train['venPresL'] < 300]
# y_train = y_train[y_train['h_index'] < 30]
