
'''
Perform supervised learning using the MPA-JHU results.
Used DR8.
'''

import numpy as np
import matplotlib.pyplot as p

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.learning_curve import plot_learning_curve
from sklearn.cross_validation import ShuffleSplit
from sklearn.svm import NuSVC

from astropy.io import fits

save_models = True
learn = True
view = True


# Load in the classifications
extra = fits.open('galSpecExtra-dr8.fits')
info = fits.opend('galSpecInfro-dr8.fits')  # has z and z_err and sn_median
line_pars = fits.open('galSpecLine-dr8.fits')

# Samples

bpt = extra[1].data["BPTCLASS"]
z = info[1].data['Z']
z_err = info[1].data['Z_ERR']
sn = info[1].data['SN_MEDIAN']

# For the line parameters, apply the same discarding process that was Used
# for the DR10 spec fits.

amps = np.hstack([line_pars[1].data["H_ALPHA_FLUX"],
                  line_pars[1].data["H_BETA_FLUX"],
                  line_pars[1].data["H_GAMMA_FLUX"],
                  line_pars[1].data["H_DELTA_FLUX"],
                  line_pars[1].data["OIII_4959_FLUX"],
                  line_pars[1].data["OIII_5007_FLUX"],
                  line_pars[1].data["NII-6584_FLUX"]])

widths = np.hstack([line_pars[1].data["H_ALPHA_EQW"],
                    line_pars[1].data["H_BETA_EQW"],
                    line_pars[1].data["H_GAMMA_EQW"],
                    line_pars[1].data["H_DELTA_EQW"],
                    line_pars[1].data["OIII_4959_EQW"],
                    line_pars[1].data["OIII_5007_EQW"],
                    line_pars[1].data["NII-6584_EQW"]])

amps_err = np.hstack([line_pars[1].data["H_ALPHA_FLUX_ERR"],
                      line_pars[1].data["H_BETA_FLUX_ERR"],
                      line_pars[1].data["H_GAMMA_FLUX_ERR"],
                      line_pars[1].data["H_DELTA_FLUX_ERR"],
                      line_pars[1].data["OIII_4959_FLUX_ERR"],
                      line_pars[1].data["OIII_5007_FLUX_ERR"],
                      line_pars[1].data["NII-6584_FLUX_ERR"]])

widths_err = np.hstack([line_pars[1].data["H_ALPHA_EQW_ERR"],
                        line_pars[1].data["H_BETA_EQW_ERR"],
                        line_pars[1].data["H_GAMMA_EQW_ERR"],
                        line_pars[1].data["H_DELTA_EQW_ERR"],
                        line_pars[1].data["OIII_4959_EQW_ERR"],
                        line_pars[1].data["OIII_5007_EQW_ERR"],
                        line_pars[1].data["NII-6584_EQW_ERR"]])

# Apply sample restrictions

# 0.01 < z < 0.26
keep = np.logical_and(z > 0.02, z < 0.26)
# z_err < 0.05
keep = np.logical_and(keep, z_err < 0.05)
# sn > 5
keep = np.logical_and(keep, sn > 5)

amps = amps[keep]
amps_err = amps_err[keep]
widths = widths[keep]
widths_err = widths_err[keep]
bpt = bpt[keep]

# Loop through the lines
for i in range(7):

    bad_amps = np.where(np.abs(amps[i]/amps_err[i]) <= 3,
                        np.isfinite(amps[i]/amps_err[i]))
    bad_widths = np.where(np.abs(widths[i]/widths_err[i]) <= 3,
                          np.isfinite(widths[i]/widths_err[i]))
    bad_errs = np.where(np.logical_or(amps_err[i] <= 0.0,
                                      widths_err[i] <= 0.0))

    amps[i, bad_amps] = 0.0
    amps[i, bad_widths] = 0.0
    amps[i, bad_errs] = 0.0
    widths[i, bad_amps] = 0.0
    widths[i, bad_widths] = 0.0
    widths[i, bad_errs] = 0.0

# Define the sets to be used
X = np.hstack([amps, widths])
y = bpt

# Finally, standardize the X data
X = (X - np.mean(X, axis=1))/np.std(X, axis=1)

# Use grid search to find optimal hyperparameters

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.5, random_state=500)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['rbf'], 'C': [1, 10, 100, 1000]}]

# Estimator
estimator = NuSVC()

# Add in a cross-validation method on top of the grid search
cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.5, random_state=500)

# Try two different scoring methods
scores = ['accuracy', 'precision', 'recall']
score = scores[0]

# Do the grid search
print("# Tuning hyper-parameters for %s" % score)

clf = GridSearchCV(estimator, tuned_parameters, cv=cv, scoring=score)
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

# Make a model with the best parameters
estimator = NuSVC(kernel='rbf', gamma=clf.best_estimator_.gamma,
                  C=clf.best_estimator_.C)

# Plot the learning curve to find a good split
title = 'NuSVC'
plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
p.show()

# Find a good number of test samples before moving on
raw_input("Continue??")

# With a good number of test samples found, fit the whole set to the model
# clf.fit(X, y)

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
