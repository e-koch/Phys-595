
'''
Perform supervised learning using the MPA-JHU results.
Used DR8.
'''

import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import NuSVC

from astropy.io import fits

save_models = True
learn = True
view = True


# Load in the classifications
extra = fits.open('galSpecExtra-dr8.fits')

line_pars = fits.open('galSpecLine-dr8.fits')

# Apply sample restrictions

y = extra[1].data["BPTCLASS"]


# For the line parameters, apply the same discarding process that was Used
# for the DR10 spec fits.

X = np.hstack([line_pars["H_ALP_AMP"]])

# Use grid search to find optimal hyperparameters

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.5, random_state=500)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['rbf'], 'C': [1, 10, 100, 1000]}]

# Try two different scoring methods
scores = ['precision', 'recall']
model = []
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)

    clf = GridSearchCV(NuSVC(), tuned_parameters, cv=5, scoring=score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print(clf.best_estimator_)
    print("Grid scores on development set:")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))

    print("Detailed classification report:")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))

    model.append(clf)

# Now take the model found, and find the outliers

outlier_percent = 0.01

## FIGURE OUT WHAT TO DO HERE!!


# Use dim reduction to look at the space.

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
