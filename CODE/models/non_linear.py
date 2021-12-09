from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

def de_tree_reg (X_train, y_train, X_val, y_val, max_depth):
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
 
    print('de_tree_reg r2:', r2_score(y_val, y_pred_val))   
    print()

    return model

def kn_reg (X_train, y_train, X_val, y_val):
    import numpy as np
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import r2_score

    pipe = Pipeline(steps = [
        ('scaler', StandardScaler()),
        ('model', KNeighborsRegressor()) ])

    model = GridSearchCV(estimator = pipe,
                        param_grid = {'model__n_neighbors': [10, 15, 20],
                                    'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                                    'model__leaf_size': [20, 30, 40]},
                        cv = 5)
    
    y_ravel = np.ravel(y_train)
    model.fit(X_train, y_ravel)
    y_pred_val = model.predict(X_val)

    print('kn_reg r2:', r2_score(y_val, y_pred_val))   
    return model

def my_svr (X_train, y_train, X_val, y_val):
    svr = SVR()
    model = svr.fit(X_train, np.ravel(y_train))
    r_sq1 = model.score(X_val, y_val)
    print('svr r2 scr:', r_sq1)
    return model

def mlp_reg (X_train, y_train, X_val, y_val):
    y_ravel = np.ravel(y_train)

    pipe = Pipeline( steps = [
                        ('scaler', StandardScaler()),
                        ('model', MLPRegressor())
    ] )
    model = GridSearchCV(estimator = pipe,
                        param_grid = {'model__alpha' : [.0001, .001, .01, .1],  #from sklearn: alpha=0.0001
                                    'model__activation' : ['tanh', 'relu'],
                                    'model__solver' : ['sgd', 'adam'],
                                    'model__learning_rate' : ['constant', 'adaptive'],
                                    'model__max_iter': [5000]
                        },
                        cv = 5
    )
    model = model.fit(X_train, y_ravel)
    y_pred_val = model.predict(X_val)

    print('mlp r2:', r2_score(y_val, y_pred_val))  
    print("mlp score:", model.score(X_val, y_val)) 
    return model


