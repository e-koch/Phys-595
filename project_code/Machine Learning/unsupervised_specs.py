
'''
Unsupervised learning on SDSS spectral data
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_validation import train_test_split
from pandas import read_csv, DataFrame
import joblib

save_models = True
learn = True
view = True

data = read_csv("")

X = None  # Some subset of the data columns

# Standardize the data
X = (X - np.mean(X, axis=1))/np.std(X, axis=1)

if learn:
    for i in range(10):
        X_train, X_test = \
            train_test_split(X, test_size=0.5, random_state=500+i)

        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(X_train)

        y_pred = clf.predict(X_test)

        if i == 0:
            anomalies = y_pred[y_pred == 1]
        else:
            anomalies = np.append(anomalies, y_pred[y_pred == 1])

        if save_models:
            joblib.dump(clf, "OneClassSVM_"+str(500+i)+"_.pkl")

    # Remove duplicated anomalies

    anomalies = np.unique(anomalies)

    anom_df = DataFrame(data.ix[anomalies])
    anom_df.to_csv("anomalies_ocsvm.csv")

if view:
    import triangle
    from dim_red_vis import dim_red

    # Use PCA to look at a projection of the set.

    subspace = dim_red(X, verbose=True)

    # Do it again with a higher dimension, then project that

    subspace = dim_red(X, n_comp=6, verbose=False)

    fig = \
        triangle.corner(subspace,
                        labels=['c'+str(i) for i in range(1, 7)])
    p.show()
