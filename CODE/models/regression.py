def simple_linear(X_train, y_train, X_val, y_val):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error

    model = LinearRegression()
    reg = model.fit(X = X_train, y = y_train)
    y_pred_val = model.predict(X_val)
    print(r2_score(y_val, y_pred_val))
    print(mean_absolute_error(y_val, y_pred_val))
    print()
    
    #return r2, mae
    

def log_reg(X_train, y_train, X_val, y_val):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    y_ravel = np.ravel(y_train)

    model = LogisticRegression(random_state = 123, max_iter = 2000)
    reg = model.fit(X = X_train_s, y = y_ravel)
    y_pred_val = model.predict(X_val_s)

    print('r2:', r2_score(y_val, y_pred_val))   # 0.006551953988217396
    print("MAE:", mean_absolute_error(y_val, y_pred_val))    # 34.07342328208346
    print()
    
def sdg_reg (X_train, y_train, X_val, y_val):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDRegressor

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_val_z  =scaler.transform(X_val)
    y_ravel = np.ravel(y_train)
    lr = [ 1, .1, .01, .001, .0001]
    settings = []
    for learning_rate in ['constant', 'optimal', 'invscaling']:
        for loss in ['squared_error', 'huber']:
            for eta0 in lr:
                model = SGDRegressor(learning_rate=learning_rate, eta0=eta0, loss=loss,random_state=666, max_iter=5000)
                model.fit(X_train_z, y_ravel)
                y_pred = model.predict(X_val_z)

                mae = mean_absolute_error(y_val, y_pred)
                r2 =  r2_score(y_val, y_pred)
                settings.append((learning_rate, eta0, loss, mae, r2))
                print(settings[-1])
