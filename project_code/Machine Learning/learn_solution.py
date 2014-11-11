
'''
Machine learning on SDSS galaxy data
'''

from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

import numpy as np


def find_params(model, data, labels, param_grid={}, test_frac=0.6, seed=500):
    '''
    Use a grid search to determine the optimum parameters for the given model.
    '''

    train_set, test_set = \
        train_test_split(data, labels, test_size=test_frac, random_state=seed)

    clf = GridSearchCV(model, param_grid)

    clf.fit(train_set)

    score = clf.score()

    pars = clf.get_params()

    return pars, score


def find_outliers(model, data, labels, params, train_frac=0.4, seed=520,
                  out_percent=0.98):
    '''
    Find outliers using the given model and data. Params should be found using
    a grid search.
    '''

    train_set, test_set = \
        train_test_split(data, labels, test_size=train_frac, random_state=seed)

    mod = model(params)

    mod.fit(train_set)

    y_pred = mod.predict(test_set)

    thresh = np.percentile(y_pred, out_percent)

    if verbose:
        print(classification_report(y_true, y_pred))


    return predict > thresh

if __name__ == "__main__":

    model = svm.OneClassSVM
