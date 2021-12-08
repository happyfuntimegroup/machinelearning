from sklearn.svm import SVC
import numpy as np

def support_vector_machine(X_train, y_train, X_val, y_val):
    svc = SVC()
    model1 = svc.fit(X_train, np.ravel(y_train))
    r_sq1 = model1.score(X_val, y_val)
    print('coefficient of determination:', r_sq1)