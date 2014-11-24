
'''
Make plots of supervised learning results.

* 3D plot of the GridSearch results (2 plots)
* BPT diagram with coloured labels for actual and predicted. (3 plots)
* PCA projection of param space with labels (2 plots)
'''

import numpy as np
from astropy.io import fits
from pandas import read_csv
import matplotlib.pyplot as p
from ast import literal_eval
from pandas import DataFrame
import seaborn as sns
sns.set()
sns.set_context('talk')

# Control which plots to make
load_speclines = False
make_gscore_img = True
make_bpt = False
make_pca = False

if load_speclines:
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

    # Load predicted labels
    bpt_pred = read_csv("supervised_prediction_labels_nusvc.csv")['0']
    bpt_pred = np.asarray(bpt_pred)

    # Finally, standardize the X data
    # X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

# Now load in the predicted labels.
# bpt_svc = read_csv()

print("Made sample set.")


# 3D plots
if make_gscore_img:
    sns.set_style('ticks')
    # Load in grid scores

    # SVC
    grid_svc = read_csv("grid_scores.csv")

    gamma_svc = []
    C_svc = []
    for dic in grid_svc['0']:
        dic = literal_eval(dic)
        gamma_svc.append(np.log10(dic['gamma']))
        C_svc.append(np.log10(dic['C']))

    gamma_svc = np.asarray(gamma_svc)
    C_svc = np.asarray(C_svc)
    # Correct for direction values go
    gamma_svc = np.unique(gamma_svc)[::-1]
    C_svc = np.round(np.unique(C_svc), 2)

    scores_svc = grid_svc['1']
    # 1st dimension should be C, second gamma
    scores_arr = scores_svc.reshape((9, 6))

    p.figure(figsize=(7, 7))
    p.imshow(scores_arr, cmap='RdBu', interpolation='nearest', vmin=0.4)

    p.xticks(np.arange(len(gamma_svc)), gamma_svc.astype(str))
    p.ylabel(r'$\mathrm{log}_{10}$ C', fontsize=16)
    p.yticks(np.arange(len(C_svc)), C_svc.astype(str))
    p.xlabel(r'$\mathrm{log}_{10}$ $\gamma$ ', fontsize=16)

    cb = p.colorbar()
    cb.set_label("Training Score", fontsize=16)

    p.savefig('svc_grid_scores.pdf')
    p.clf()

    # NuSVC
    grid_nu = read_csv("grid_scores_nusvc.csv")
    gamma_nu = []
    nu_nu = []
    for dic in grid_nu['0']:
        dic = literal_eval(dic)
        gamma_nu.append(np.log10(dic['gamma']))
        nu_nu.append(np.log10(dic['nu']))

    gamma_nu = np.asarray(gamma_nu)
    nu_nu = np.asarray(nu_nu)
    # Correct for direction values go
    gamma_nu = np.unique(gamma_nu)
    nu_nu = np.round(np.unique(nu_nu), 2)

    scores_nu = grid_nu['1']
    # 1st dimension should be C, second gamma
    scores_arr = scores_nu.reshape((6, 6))

    p.imshow(scores_arr, cmap='RdBu', interpolation='nearest')

    p.xticks(np.arange(len(gamma_nu)), gamma_nu.astype(str))
    p.ylabel(r'$\mathrm{log}_{10}$ $\nu$', fontsize=16)
    p.yticks(np.arange(len(nu_nu)), nu_nu.astype(str))
    p.xlabel(r'$\mathrm{log}_{10}$ $\gamma$', fontsize=16)

    cb = p.colorbar()
    cb.set_label("Training Score", fontsize=16)

    p.savefig('nusvc_grid_scores.pdf')

# BPT Diagrams
if make_bpt:
    execfile("/Users/eric/Dropbox/Phys 595/Phys-595/project_code/Machine Learning/hist2d.py")
    # Make sure data is loaded in.
    assert load_speclines

    # Define the demarcations for the BPT diagram
    blue = lambda x: 0.61 / (x - 0.05) + 1.3
    red = lambda x: 0.61 / (x - 0.47) + 1.19

    # Data
    y_bpt = np.log10(amps[:, 5]/amps[:, 1])
    x_bpt = np.log10(amps[:, 6]/amps[:, 0])
    df = DataFrame([x_bpt, y_bpt, bpt_pred]).T
    df = df.replace([np.inf, -np.inf], np.NaN)
    df = df.dropna()

    cmaps = ['binary', 'Blues', 'Greens', 'Oranges', 'Purples', 'Reds']
    colors = ['k', 'b', 'g', 'orange', 'purple', 'r']
    labels = ["SF", "Low S/N SF", "Comp", "AGN", "Low S/N AGN"]
    bins = [100, 100, 50, 60, 100]

    p.figure(figsize=(8, 8))

    for i, lab in enumerate(np.unique(bpt)[1:]):
        if lab == -1:
            continue
        subset = df[df[2] == lab]
        kwywrds = {'cmap': cmaps[i], 'bins': bins[i], 'color': colors[i]}
        hist2d(np.asarray(subset[0]),
               np.asarray(subset[1]), **kwywrds)
    p.xlim([-1.8, 0.5])
    p.ylim([-1.1, 1.5])

    p.xlabel(r'$\mathrm{log}_{10}$ [NII]5584/H$\alpha$')
    p.ylabel(r'$\mathrm{log}_{10}$ [OIII]5007/H$\beta$')

    # Plot the lines
    x_vals = np.linspace(-1.8, 0.3, 100)
    p.plot(x_vals[:-20], blue(x_vals[:-20]), 'b-')
    p.plot(x_vals, red(x_vals), 'r-')

    p.legend([p.Rectangle((0, 0), 1, 1, fc=col) for col in colors[:-1]], labels, loc='best')
    p.show()

# PCA projection
if make_pca:
    pass
