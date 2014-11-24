
'''
Quick dim reduction for visualization.
'''

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as p


def dim_red(X, n_comp=2, verbose=True):
    '''
    Applies a dimensionality reduction
    '''

    # Subtract the mean off each column

    X -= np.mean(X, axis=0)

    mod = PCA(n_components=n_comp)

    mod.fit(X)

    subspace = mod.transform(X)

    if verbose:
        if n_comp <= 2:
            p.scatter(subspace[:, 0], subspace[:, 1])
        elif n_comp == 3:
            pass
        else:
            print("Too high of a dimension to view.")
        p.show()

    return subspace

## port in yt to make visualization of physical data comparisons.
