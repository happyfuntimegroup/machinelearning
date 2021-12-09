def simple_linear(X_train, y_train, X_val, y_val):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error

    model = LinearRegression()
    reg = model.fit(X = X_train, y = y_train)
    y_pred_val = reg.predict(X_val)
    print("LinearRegression r2:", r2_score(y_val, y_pred_val))
    print("LinearRegression MAE:", mean_absolute_error(y_val, y_pred_val))
    print()
    
    return reg
    

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
    y_pred_val = reg.predict(X_val_s)

    print('log_reg r2:', r2_score(y_val, y_pred_val))   # 0.006551953988217396
    print("log_reg MAE:", mean_absolute_error(y_val, y_pred_val))    # 34.07342328208346
    print()
    
def sdg_reg (X_train, y_train, X_val, y_val):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import r2_score, mean_absolute_error

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
                reg = model.fit(X_train_z, y_ravel)
                y_pred = reg.predict(X_val_z)

                mae = mean_absolute_error(y_val, y_pred)
                r2 =  r2_score(y_val, y_pred)
                settings.append((learning_rate, eta0, loss, mae, r2))
                print("sdg_reg r2:", settings[-1])

def poly_reg (X_train, y_train, X_val, y_val, degree):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    #from sklearn.linear_model import LinearRegression   
    from sklearn.linear_model import Ridge # Try this with ridge instead?
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_val_z  =scaler.transform(X_val)

    polynomial_features = PolynomialFeatures(degree = degree)
    x_train_poly = polynomial_features.fit_transform(X_train_z)
    x_val_poly = polynomial_features.transform(X_val_z)

    #model = LinearRegression()
    model = Ridge(alpha = 1.0)
    reg = model.fit(x_train_poly, y_train)
    y_poly_pred = reg.predict(x_val_poly)

    print("poly_reg r2:", r2_score(y_val, y_poly_pred))   # -0.04350391168707901
    print("poly_reg MAE", mean_absolute_error(y_val, y_poly_pred))    # 32.65668266590838

    #source: https://towardsdatascience.com/polynomial-regression-bbe8b9d97491

def pois_reg (X_train, y_train, X_val, y_val):
    import numpy as np
    from sklearn.linear_model import PoissonRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    y_ravel = np.ravel(y_train)

    model = PoissonRegressor()
    reg = model.fit(X = X_train_s, y = y_ravel)
    y_pred_val = reg.predict(X_val_s)

    print('pois_reg r2:', r2_score(y_val, y_pred_val))
    print("pois_reg MAE:", mean_absolute_error(y_val, y_pred_val))