from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

def support_vector_machine(X_train, y_train, X_val, y_val):
    svc = SVC()
    model = svc.fit(X_train, np.ravel(y_train))
    r2 = model.score(X_val, y_val)
    print('r2', r2)

    return model


def decision_tree(X_train, y_train, X_val, y_val):
    """
    This was a very early attempt at a decision tree. We never got it properly working.
    It convinced us that we should be looking for numerical features and related models.
    """
    X, y = X_train['venue'], X_train['citations']
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    le_venue = LabelEncoder()
    le_topics = LabelEncoder()

    X_train['venue_n'] = le_venue.fit_transform(X_train['venue'].astype(str))
    X_train['topics_n'] = le_topics.fit_transform(X_train['topics'].astype(str))

    X_train_new = X_train.drop(['doi', 'title', 'references', 'year', 'topics', 'venue', 'abstract', 'authors', 'fields_of_study', 'is_open_access'],axis='columns')

    model = tree.DecisionTreeClassifier()
    clas = model.fit(X_train_new, y_train)
    y_pred_val = clas.predict(X_val)
    print("DecisionTreeClassifier:", r2_score(y_val, y_pred_val))

    return model