
'''
Fit Gaussian models to the Hydrogen Balmer series for SDSS galaxy spectra.
'''

import numpy as np
from pyspeckit import Spectrum
from pyspeckit.mpfit.mpfit import mpfitException
from astropy.io import fits
from pandas import Series
from scipy.ndimage import median_filter


line_dict = {"Halp + NII": [None, 6548.0, 3.0, None, 6562.8, 3.0, None, 6583.4, 3.0],
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
               fix_lambda=True, smooth_size=3, verbose=False,
               verbose_print=True):
    '''
    Given a FITS file, fit the specified lines to the spectra.
    '''

    if verbose_print:
        print "Running on " + filename

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

        spec_line = spec.slice(start=line_props[1]-100,
                               stop=line_props[-2]+100,
                               units='Angstroms')

        if verbose:
            spec_line.plotter()

        spec_line.baseline(order=1, subract=False, annotate=False)

        # Estimate the amplitude of the line after background subtraction
        # using the peak of where the line is expected to be
        baseline_model = lambda x, p: p[1] + p[0]*x

        base_params = spec_line.baseline.baselinepars

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

        try:
            spec_line.specfit(guesses=line_props, fixed=fix,
                              fittype="gaussian",
                              multifit=multifit)

            line_pars = spec_line.specfit.modelpars
            line_errors = spec_line.specfit.modelerrs

        except ValueError:
            line_pars = [0.0] * 3 * num_lines
            line_errors = [0.0] * 3 * num_lines

        except mpfitException:
            line_pars = [np.NaN] * 3 * num_lines
            line_errors = [np.NaN] * 3 * num_lines

        # For multifits, a bad fit for a line that isn't there increases the
        # error on a fit for a line that is. Remove
        # if num_lines > 1:
        #     t_stats = [np.logical_and(np.abs(k)/j < 1, j != 0)
        #                for k, j in zip(line_pars, line_errors)]

        #     if np.any(t_stats):
        #         posn = [f for f, j in enumerate(t_stats) if j]
        #         bad_line = []
        #         for pos in posn:
        #             for n in range(1, num_lines+1):
        #                 if pos < 3*i and pos > 3*(i-1):
        #                     bad_line.append(n)
        #                     break

        #         # If they're both bad, just continue
        #         if len(bad_line) == num_lines:
        #             pass

        #         else:
        #             bad_line = np.sort(bad_line)[::-1]
        #             for bad in bad_line:
        #                 bad = int(bad) - 1
        #                 for p in range(3)[::-1]:
        #                     line_props.pop(3*bad + p)
        #                     fix.pop(3*bad + p)

        #             if (num_lines - len(bad_line)) == 1:
        #                 multifit = False

        #             spec_line.specfit(guesses=line_props,
        #                               fixed=fix,
        #                               fittype="gaussian",
        #                               multifit=multifit)

        #             good_line = \
        #                 np.asarray(list(set(range(1, num_lines+1)) & set(bad_line)))

        #             good_line -= 1

        #             for good in good_line:
        #                 line_pars[3*(good-1):3*good] = spec_line.specfit.modelpars
        #                 line_errors[3*(good-1):3*good] = spec_line.specfit.modelerrs

        line_params.extend(line_pars)
        line_errs.extend(line_errors)

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
