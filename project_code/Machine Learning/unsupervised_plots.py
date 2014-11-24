
'''
Plots for unsupervised results.
'''

import glob
import numpy as np
from astropy.io import fits
from pandas import read_csv
import matplotlib.pyplot as p
from scipy.ndimage import gaussian_filter1d
from pandas import DataFrame
import seaborn as sns
sns.set()
sns.set_context('talk')
# sns.set_style('ticks')


make_gridsearch = False
make_pca = True
classify_anomspecs = False
make_anomspecs = False

if make_gridsearch:
    # SVC
    grid_svc = read_csv("svm_anomaly_testing_fifth.csv")

    gamma_svc = np.log10(grid_svc["Nu"])
    nu_svc = np.log10(grid_svc['Gamma'])
    percents = np.log10(grid_svc["Percent"])

    gamma_svc = np.asarray(gamma_svc)
    nu_svc = np.asarray(nu_svc)
    # Correct for direction values go
    gamma_svc = np.round(np.unique(gamma_svc)[::-1], 2)
    nu_svc = np.round(np.unique(nu_svc), 2)

    # 1st dimension should be C, second gamma
    percents = np.asarray(percents).reshape((4, 4))

    p.imshow(percents, cmap='RdBu', interpolation='nearest')

    p.xticks(np.arange(len(gamma_svc)), gamma_svc.astype(str))
    p.ylabel(r'$\mathrm{log}_{10}$ $\nu$', fontsize=16)
    p.yticks(np.arange(len(nu_svc)), nu_svc.astype(str))
    p.xlabel(r'$\mathrm{log}_{10}$ $\gamma$', fontsize=16)

    cb = p.colorbar()
    cb.set_label(r"$\log_{10}$ $\mathrm{Anomaly}$ $\mathrm{Percent}$", fontsize=16)

    p.show()
    raw_input("Continue?")

    # LSQ
    grid_lsq = read_csv("lsq_anomaly_testing_fifth_betterparams.csv")

    # Take only the interesting grid values.
    grid_lsq = grid_lsq[grid_lsq["Rho"] > 1e-3]
    grid_lsq = grid_lsq[grid_lsq["Rho"] < 52]
    grid_lsq = grid_lsq[grid_lsq["Sigma"] > 7]
    grid_lsq = grid_lsq[grid_lsq["Sigma"] < 370]

    sigma = np.log10(grid_lsq["Sigma"])
    rho = np.log10(grid_lsq["Rho"])

    sigma = np.asarray(sigma)
    rho = np.asarray(rho)

    sigma = np.round(np.unique(sigma), 2)
    rho = np.round(np.unique(rho), 2)

    percents = np.asarray(grid_lsq["Percent"]).reshape((4,4))
    percents = np.log10(percents)

    p.imshow(percents, cmap='RdBu', interpolation='nearest')

    p.xticks(np.arange(len(sigma)), sigma.astype(str))
    p.ylabel(r'$\mathrm{log}_{10}$  $\sigma$', fontsize=16)
    p.yticks(np.arange(len(rho)), rho.astype(str))
    p.xlabel(r'$\mathrm{log}_{10}$  $\rho$ ', fontsize=16)

    cb = p.colorbar()
    cb.set_label(r"$\log_{10}$ $\mathrm{Anomaly}$ $\mathrm{Percent}$",
                 fontsize=16)

    p.show()
if make_pca:
    import triangle
    from dim_red_vis import dim_red
    execfile("/Users/eric/Dropbox/Phys 595/Phys-595/project_code/Machine Learning/hist2d.py")

    data = read_csv("all_spec_data_cleaned.csv")

    X = data[data.columns[1:]]
    X = np.asarray(X)

    # Standardize the data
    X = (X - np.mean(X, axis=0))/np.std(X, axis=0)

    # Use PCA to look at a projection of the set.

    subspace = dim_red(X, verbose=False)

    hist2d(subspace[:, 0], subspace[:, 1], **{'bins': 10})

    # Overplot anomalies on the space
    svc_anoms = np.asarray(read_csv("svc_anom_cut.csv")["Unnamed: 0.1"])
    p.scatter(subspace[svc_anoms, 0], subspace[svc_anoms, 1], c='r', s=30,
              label="SVC Anomalies")

    lsq_anoms = np.unique(read_csv("anomalies_lsq.csv")["Unnamed: 0"])
    p.scatter(subspace[lsq_anoms, 0], subspace[lsq_anoms, 1], c='b', s=30,
              label="LSQ Anomalies")

    both_anoms = np.asarray(list(set(svc_anoms) & set(lsq_anoms)))
    p.scatter(subspace[both_anoms, 0], subspace[both_anoms, 1], c='g', s=30,
              label="Both")

    p.xlabel("PC 1", fontsize=16)
    p.ylabel("PC 2", fontsize=16)

    p.xlim((-80, 30))
    p.ylim((-150, 10))
    p.legend(loc='best', fontsize=16)
    p.show()

    # Do it again with a higher dimension, then project that

    # subspace = dim_red(X, n_comp=6, verbose=False)

    # fig = \
    #     triangle.corner(subspace,
    #                     labels=['c'+str(i) for i in range(1, 7)])
    # p.show()

if classify_anomspecs:
    spectra_folder = "lsq_anomalies/"

    spectra = glob.glob(spectra_folder+"*")

    classes = np.empty(len(spectra))

    for i, spec in enumerate(spectra):
        spec_file = fits.open(spec)

        flux = spec_file[1].data["flux"]
        smooth = gaussian_filter1d(flux, 3)
        lam_wav = 10**spec_file[1].data["loglam"] / \
            (1 + spec_file[2].data["Z"])

        # Cut down to the area of interest
        smooth = smooth[np.logical_and(lam_wav > 3700, lam_wav < 6800)]
        flux = flux[np.logical_and(lam_wav > 3700, lam_wav < 6800)]
        lam_wav = lam_wav[np.logical_and(lam_wav > 3700, lam_wav < 6800)]

        print spec
        p.plot(lam_wav, smooth)
        p.plot(lam_wav, flux, alpha=0.2)
        p.show()
        # classes[i] = int(raw_input("Class?"))
        raw_input("??")
        p.cla()

    df = DataFrame(np.vstack([spectra, classes]).T,
                   columns=["Spec Names", "Anomaly Classes"])

if make_anomspecs:
    # Load in classifications and plot an example of each

    classes_lsq = read_csv("lsq_anomaly_classification.csv")

    posn = [0, 2, 0, 1, 2, 0]

    f, axarr = p.subplots(3, 2)
    grid_posns = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    titles = ["1", "2", "3", "4", "5", "6"]

    for i in range(1, 7):
        filename = classes_lsq["Spec Names"][classes_lsq["Anomaly Classes"] == i].iloc[posn[i-1]]
        print filename

        spec_file = fits.open(filename)

        flux = spec_file[1].data["flux"]
        smooth = gaussian_filter1d(flux, 3)
        lam_wav = 10**spec_file[1].data["loglam"] / \
            (1 + spec_file[2].data["Z"])

        # Cut down to the area of interest
        smooth = smooth[np.logical_and(lam_wav > 3700, lam_wav < 6800)]
        lam_wav = lam_wav[np.logical_and(lam_wav > 3700, lam_wav < 6800)]

        axarr[grid_posns[i-1][0], grid_posns[i-1][1]].plot(lam_wav, smooth, linewidth=0.75)
        axarr[grid_posns[i-1][0], grid_posns[i-1][1]].set_title(titles[i-1])
        # p.show()
        # raw_input("Continue?")
        # p.cla()
    # Add labels
    f.text(0.5, 0.03, r"Wavelength ($\AA$)", ha='center')
    f.text(0.04, 0.5, r"Flux ($10^{-17}$ erg/s/cm$^2$/$\AA$)", va='center', rotation='vertical')
    p.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    p.setp([a.get_xticklabels() for a in axarr[1, :]], visible=False)
    p.autoscale(True)
    p.show()
