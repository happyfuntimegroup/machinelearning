def de_tree_reg (X_train, y_train, X_val, y_val, max_depth):
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import r2_score
    import numpy as np

#    scaler = StandardScaler()
    pipe = Pipeline( steps = [
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(max_depth=max_depth))
    ] )
    model = GridSearchCV(estimator = pipe,
                        param_grid = {'model__criterion': ['squared_error', 'absolute_error', 'poisson']},
                        cv = 5
    )
    model.fit(X_train, np.ravel(y_train))
    y_pred_val = model.predict(X_val)
#    X_train_z = scaler.fit_transform(X_train)
#    X_val_z  =scaler.transform(X_val)
#
#    model = RandomForestRegressor(max_depth = max_depth, random_state = 123)  # Other parameters?
#    reg = model.fit(X_train_z, y_train)
#    y_pred_val = reg.predict(X_val_z)

    print('r2:', r2_score(y_val, y_pred_val))   
    print()


def kn_reg (X_train, y_train, X_val, y_val, neighbors, algorithm, leaf_sz):
    import numpy as np
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_val_z  =scaler.transform(X_val)

    y_ravel = np.ravel(y_train)

    model = KNeighborsRegressor(n_neighbors=neighbors, algorithm = algorithm, leaf_size=leaf_sz)
    reg = model.fit(X_train_z, y_ravel)
    y_pred_val = reg.predict(X_val_z)

    print('r2:', r2_score(y_val, y_pred_val))   
    print()


def my_svr (X_train, y_train, X_val, y_val):
    from sklearn.svm import SVR
    import numpy as np
    svr = SVR()
    model1 = svr.fit(X_train, np.ravel(y_train))
    r_sq1 = model1.score(X_val, y_val)
    print('r2 scr:', r_sq1)

def mlp_reg (X_train, y_train, X_val, y_val, maxit=500, activation='relu', solver='adam', alpha=0.0001, lr='constant'):
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    y_ravel = np.ravel(y_train)

    pipe = Pipeline( steps = [
                        ('scaler', StandardScaler()),
                        ('model', MLPRegressor())
    ] )
    model = GridSearchCV(estimator = pipe,
                        param_grid = {'model__activation' : ['tanh', 'relu'],
                                    'model__solver' : ['sgd', 'adam'],
                                    'model__learning_rate' : ['constant', 'adaptive'],
                                    'model__max_iter': [500]
                        },
                        cv = 5
    )
#    model = MLPRegressor(max_iter = maxit, activation = activation, solver=solver, alpha = alpha, learning_rate = lr, random_state = 123)
    reg = model.fit(X_train, y_ravel)
    y_pred_val = reg.predict(X_val)

    print('r2:', r2_score(y_val, y_pred_val))  
    print("score:", reg.score(X_val, y_val)) 
    print()



