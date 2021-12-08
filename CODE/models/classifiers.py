from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder

def support_vector_machine(X_train, y_train, X_val, y_val):
    svc = SVC()
    model1 = svc.fit(X_train, np.ravel(y_train))
    r_sq = model1.score(X_val, y_val)
    print('coefficient of determination:', r_sq)

    return r_sq



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

    inputs = X_train.drop('citations', axis='columns')
    target = X_train['citations']

    inputs['venue_n'] = le_venue.fit_transform(inputs['venue'].astype(str))
    inputs['topics_n'] = le_topics.fit_transform(inputs['topics'].astype(str))

    inputs_n = inputs.drop(['doi', 'title', 'references', 'year', 'topics', 'venue', 'abstract', 'authors', 'fields_of_study', 'is_open_access'],axis='columns')

    model = tree.DecisionTreeClassifier()
    model = model.fit(X_train, y_train)

