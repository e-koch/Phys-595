
'''
Unsupervised learning on SDSS spectral data
'''

import numpy as np
import matplotlib.pyplot as p
from sklearn import svm
from sklearn.cross_validation import train_test_split
from pandas import read_csv, DataFrame
import joblib

save_models = True
test_params = True
learn = False
view = False

data = read_csv("all_spec_data_cleaned.csv")

X = data[data.columns[1:]]
X = np.asarray(X)

# Standardize the data
X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

print "Ready for some anomaly finding!"

if test_params:
    # Make a mock Grid Search, evaluating based on the number of
    # anomalies found
    nus = np.linspace(0.01, 1.0, 7)
    gammas = np.linspace(1e-4, 0.5, 7)

    # Use a subset of the data to speed things up a bit
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_sub = X[indices[:len(indices)/2]]

    X_train, X_test = \
        train_test_split(X, test_size=0.5, random_state=300)

    results = []

    for nu in nus:
        for gamma in gammas:
            print "Now fitting: Nu - %s; Gamma - %s" % (nu, gamma)
            clf = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=gamma,
                                  verbose=True)
            clf.fit(X_train)

            y_pred = clf.predict(X_test)
            anomalies = np.where(y_pred == -1)[0]

            results.append([nu, gamma, len(anomalies)])

    test_df = DataFrame(results, columns=["Nu", "Gamma", "Anomalies"])
if learn:
    for i in range(1):
        print "On %s/%s" % (i, 10)
        X_train, X_test = \
            train_test_split(X, test_size=0.5, random_state=500+i)

        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1, verbose=True)
        clf.fit(X_train)

        y_pred = clf.predict(X_test)

        if i == 0:
            anomalies = np.where(y_pred == -1)[0]
        else:
            anomalies = np.append(anomalies, np.where(y_pred == -1)[0])

        if save_models:
            joblib.dump(clf, "OneClassSVM_"+str(500+i)+".pkl")

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
