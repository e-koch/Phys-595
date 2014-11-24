
'''
Unsupervised learning on SDSS spectral data
'''

import numpy as np
import matplotlib.pyplot as p
from sklearn.cross_validation import train_test_split
from pandas import read_csv, DataFrame, concat
import joblib

from lsanomaly import LSAnomaly

save_models = False
test_params = False
multi = True
learn = True
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
    sigmas = np.logspace(0, 3, 8)
    rhos = np.logspace(-6, 3, 8)

    # Use a subset of the data to speed things up a bit
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_sub = X[indices[:len(indices)/5]]

    X_train, X_test = \
        train_test_split(X, test_size=0.5, random_state=200)

    def testing_func(a):
        sigma, rho = a
        print "Training with rho: %s, sigma: %s" % (rho, sigma)
        clf = LSAnomaly(rho=rho, sigma=sigma)
        clf.fit(X_train)

        y_pred = clf.predict(X_test)
        y_pred = np.asarray(y_pred)
        anomalies = np.where(y_pred == 'anomaly')[0]
        print anomalies

        return [sigma, rho, len(anomalies),
                len(anomalies)/float(X_test.shape[0])]

    if multi:
        from multiprocessing import Pool
        from itertools import product

        pool = Pool(processes=8)
        results = pool.map(testing_func, product(sigmas, rhos))

        pool.close()
        pool.join()
    else:
        results = []

        for rho in rhos:
            for sigma in sigmas:
                print "Now fitting: rho - %s; sigma - %s" % (rho, sigma)
                clf = LSAnomaly(rho=rho, sigma=sigma, verbose=False)
                clf.fit(X_train)

                y_pred = clf.predict(X_test)
                anomalies = np.where(y_pred == 'anomaly')[0]
                print np.unique(anomalies)

                results.append([nu, gamma, len(anomalies),
                                len(anomalies)/float(X_test.shape[0])])

    test_df = DataFrame(results, columns=["Sigma", "Rho", "Anomalies",
                                          "Percent"])
    test_df.to_csv("lsq_anomaly_testing_fifth_betterparams.csv")
if learn:

    all_anom = []
    # Repeat the process many times
    # Record all anomalies and look for those which are consistently
    # labeled.
    for i in range(100):
        print "On %s/%s" % (i, 100)
        # Need to keep track of the indices!
        indices = np.arange(X.shape[0])
        X_train, X_test, ind_train, ind_test = \
            train_test_split(X, indices, test_size=0.5,
                             random_state=np.random.randint(1e8))

        clf = LSAnomaly(rho=2.7, sigma=51.8)
        clf.fit(X_train)

        y_pred = clf.predict(X)
        y_pred = np.asarray(y_pred)
        anomalies = np.where(y_pred == 'anomaly')[0]

        all_anom.append(DataFrame(data.ix[anomalies]))

        if save_models:
            joblib.dump(clf, "OneClassSVM_"+str(500+i)+".pkl")
        del clf

    anom_df = concat(all_anom)
    anom_df.to_csv("anomalies_lsq.csv")

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
