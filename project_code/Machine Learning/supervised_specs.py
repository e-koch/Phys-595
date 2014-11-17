
'''
Perform supervised learning using the MPA-JHU results.
Used DR8.
'''

# Setup non-interactive plotting
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as p

# Use seaborn for pretty plots
import seaborn

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from plot_learning_curve import plot_learning_curve
from sklearn.cross_validation import ShuffleSplit
from sklearn.svm import SVC

from astropy.io import fits

save_models = True
learn = True
view = True


# Load in the classifications
extra = fits.open('galSpecExtra-dr8.fits')
info = fits.open('galSpecInfo-dr8.fits')  # has z and z_err and sn_median
line_pars = fits.open('galSpecLine-dr8.fits')

# Samples

bpt = extra[1].data["BPTCLASS"]
z = info[1].data['Z']
z_err = info[1].data['Z_ERR']
sn = info[1].data['SN_MEDIAN']

# For the line parameters, apply the same discarding process that was Used
# for the DR10 spec fits.

amps = np.vstack([line_pars[1].data["H_ALPHA_FLUX"],
                  line_pars[1].data["H_BETA_FLUX"],
                  line_pars[1].data["H_GAMMA_FLUX"],
                  line_pars[1].data["H_DELTA_FLUX"],
                  line_pars[1].data["OIII_4959_FLUX"],
                  line_pars[1].data["OIII_5007_FLUX"],
                  line_pars[1].data["NII_6584_FLUX"]]).T

widths = np.vstack([line_pars[1].data["H_ALPHA_EQW"],
                    line_pars[1].data["H_BETA_EQW"],
                    line_pars[1].data["H_GAMMA_EQW"],
                    line_pars[1].data["H_DELTA_EQW"],
                    line_pars[1].data["OIII_4959_EQW"],
                    line_pars[1].data["OIII_5007_EQW"],
                    line_pars[1].data["NII_6584_EQW"]]).T

amps_err = np.vstack([line_pars[1].data["H_ALPHA_FLUX_ERR"],
                      line_pars[1].data["H_BETA_FLUX_ERR"],
                      line_pars[1].data["H_GAMMA_FLUX_ERR"],
                      line_pars[1].data["H_DELTA_FLUX_ERR"],
                      line_pars[1].data["OIII_4959_FLUX_ERR"],
                      line_pars[1].data["OIII_5007_FLUX_ERR"],
                      line_pars[1].data["NII_6584_FLUX_ERR"]]).T

widths_err = np.vstack([line_pars[1].data["H_ALPHA_EQW_ERR"],
                        line_pars[1].data["H_BETA_EQW_ERR"],
                        line_pars[1].data["H_GAMMA_EQW_ERR"],
                        line_pars[1].data["H_DELTA_EQW_ERR"],
                        line_pars[1].data["OIII_4959_EQW_ERR"],
                        line_pars[1].data["OIII_5007_EQW_ERR"],
                        line_pars[1].data["NII_6584_EQW_ERR"]]).T

# Close data files

extra.close()
info.close()
line_pars.close()

print("Loaded data. Starting restriction...")

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
                        np.isfinite(amps[i]/amps_err[i]), 1)
    bad_widths = np.where(np.abs(widths[i]/widths_err[i]) <= 3,
                          np.isfinite(widths[i]/widths_err[i]), 1)
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
X = (X - np.mean(X, axis=0))/np.std(X, axis=0)
# Keep a copy of the entire data set
X_all = X.copy()
y_all = y.copy()
# Unfortunately the method cannot handle the size of the dataset
# Test on a randomly selected sample using half of the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices[:len(indices)/2]]
y = y[indices[:len(indices)/2]]

print("Made sample set. Starting grid search.")

# Use grid search to find optimal hyperparameters

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.5, random_state=500)

# Set the parameters by cross-validation
tuned_parameters = [{'gamma': [0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                     'C': [0.1, 0.5, 1, 5, 10, 50, 100, 250, 500]}]

# Estimator
estimator = SVC(kernel='rbf', cache_size=2000, class_weight='auto')

# Add in a cross-validation method on top of the grid search
cv = ShuffleSplit(X_train.shape[0], n_iter=3, test_size=0.8, random_state=500)

# Try two different scoring methods
scores = ['accuracy', 'precision', 'recall']
score = scores[0]

# Do the grid search
print("# Tuning hyper-parameters for %s" % score)

clf = GridSearchCV(estimator, tuned_parameters, cv=cv, scoring=score, n_jobs=4,
                   verbose=2)
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
estimator = SVC(kernel='rbf', gamma=clf.best_estimator_.gamma,
                C=clf.best_estimator_.C)

# Plot the learning curve to find a good split
title = 'SVC'
plot_learning_curve(estimator, title, X_train, y_train, cv=cv, n_jobs=4)
p.savefig("supervised_learning.pdf")

# Find a good number of test samples before moving on
raw_input("Continue??")

# With a good number of test samples found, fit the whole set to the model
estimator.fit(X_all, y_all)
y_pred = estimator.predict(X_all, y_all)
print(classification_report(y_all, y_pred))

# Hold here
raw_input("Continue??")


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
