
'''
Fit Gaussian models to the Hydrogen Balmer series for SDSS galaxy spectra.
'''

import numpy as np
from pyspeckit import Spectrum
from astropy.io import fits
from pandas import Series
from scipy.ndimage import median_filter


line_dict = {"Halp + NII": [None, 6562.8, 3.0, None, 6583.4, 3.0],
             "Hbet": [None, 4861.3, 3.0],
             "Hgam": [None, 4340.0, 3.0],
             "Hdel": [None, 4102.8, 3.0],
             "Ca H + Ca K": [None, 3933.7, 3.0, None, 3968.5, 3.0],
             "Mg": [None, 5175.3, 3.0],
             "NaI": [None, 5894.0, 3.0],
             "OIIIa + OIIIb": [None, 4959.0, 3.0, None, 5006.8, 3.0]}


def do_specfit(filename, lines=["Halp + NII", "Hbet", "Hgam", "Hdel",
                                "Ca H + Ca K",
                                "Mg", "NaI", "OIIIa + OIIIb"],
               fix_lambda=True, smooth_size=3, verbose=False):
    '''
    Given a FITS file, fit the specified lines to the spectra.
    '''

    spec_file = fits.open(filename)

    flux = spec_file[1].data["flux"]

    # Correct for redshift
    lam_wav = 10**spec_file[1].data["loglam"] / (1 + spec_file[2].data["Z"])

    # Correct for vacuum-to-air
    lam_wav = vac_to_air(lam_wav)

    # Optionally smooth the spectrum
    if smooth_size != 0:
        flux = median_filter(flux, smooth_size)

    spec = Spectrum(data=flux, xarr=lam_wav,
                    header=spec_file[1].header,
                    unit="",
                    xarrkwargs={'unit': 'Angstroms'})

    line_params = []
    line_errs = []

    for i, line in enumerate(lines):

        line_props = line_dict[line]
        num_lines = len(line_props) / 3

        # if verbose:
        spec.plotter(xmin=line_props[1]-100, xmax=line_props[-2]+100)

        spec.baseline(xmin=line_props[1]-100, xmax=line_props[-2]+100,
                      exclude=[[line_props[1]-20, line_props[-2]+20]], order=1,
                      subtract=False, highlight_fitregions=False,
                      reset_selection=True, annotate=False)

        # Estimate the amplitude of the line after background subtraction
        # using the peak of where the line is expected to be
        baseline_model = lambda x, p: p[1] + p[0]*x

        base_params = spec.baseline.baselinepars

        for i in range(num_lines):
            lam_line = find_nearest(lam_wav, line_props[3*i+1])

            line_props[3*i] = float(flux[np.argwhere(lam_wav == lam_line)]) - \
                baseline_model(lam_line, base_params)

        if fix_lambda:
            fix = [False, True, False] * num_lines
        else:
            fix = [False] * len(line_props)

        if num_lines > 1:
            multifit = True
        else:
            multifit = False

        spec.specfit(guesses=line_props, fixed=fix, fittype="gaussian",
                     multifit=multifit)

        line_params.extend(spec.specfit.modelpars)
        line_errs.extend(spec.specfit.modelerrs)

        if verbose:
            raw_input("Continue?")

    rows = len(line_params) / 3
    line_params = np.array(line_params).reshape((rows, 3))
    line_errs = np.array(line_errs).reshape((rows, 3))

    # No point in saving lambda if it is fixed.
    if fix_lambda:
        line_params = np.hstack([line_params[:, 0], line_params[:, -1]])
        line_errs = np.hstack([line_errs[:, 0], line_errs[:, -1]])
        line_param_names = ['Amplitude', 'Width', 'Amplitude Error',
                            'Width Error']

    line_names = []
    for line in lines:
        line = line.split(" + ")
        if len(line) > 1:
            for l in line:
                line_names.append(l)
        else:
            line_names.append(line[0])

    line_and_par_names = []
    for par in line_param_names:
        for name in line_names:
            line_and_par_names.append(name+" "+par)

    # Return as a named series, which can be concatenated into a dataframe.
    ser = Series(np.hstack([line_params, line_errs]).ravel(),
                 index=line_and_par_names)

    return ser


# Utility Functions

def find_nearest(array, value):
    '''
    From:
    http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    '''
    idx = (np.abs(array-value)).argmin()
    return array[idx]


@np.vectorize
def vac_to_air(lam):
    '''
    Vacuum to air wavelength conversion from Morton (1991)
    '''
    return lam / (1.0 + 2.735182e-4 + 131.4182 * lam**-2 + 2.76249e8 * lam**-4)
